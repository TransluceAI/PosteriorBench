"""A run of the prompt family on one task and replicate: prompt selection and mixtures at each
pool size (`selection`), Instruction, and their compositions. The context gives the task's
original model, response limit and labels, for package and generalization tasks alike.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from posteriorbench import data, instructions, ledger

from posteriorbench_methods.common import charges, composition, rank_table, run_context, run_record
from posteriorbench_methods.generalization import instructions as generalization_instructions
from posteriorbench_methods.generalization import tasks as generalization_tasks
from posteriorbench_methods.prompt import (
    draws,
    methods,
    pool,
    samplers,
    selection,
    settings,
    shared,
)


def check(options: run_context.RunOptions, settings: settings.Settings) -> None:
    """Raise before the model is loaded: methods.MethodError if no method is left to run,
    pool.PoolError if a method needs the pool file and none was given."""
    asked = asked_methods(settings.methods, compose=options.compose)
    if not asked:
        raise methods.MethodError(
            "no method to run: `methods` is empty, or it holds only names with +composition "
            "and --no-compose leaves those out"
        )
    reads_pool = any(not isinstance(method, methods.Instruction) for method in asked)
    if reads_pool and settings.pool_sizes:
        pool.pool_file(options.method_data)
    elif methods.Instruction() not in asked:
        raise methods.MethodError(
            "no method to run: prompt selection and mixtures need a pool size, and `pool_sizes` "
            "is empty"
        )


def run(
    context: run_context.RunContext, settings: settings.Settings
) -> list[run_record.RunSummary]:
    """Run `settings.methods` on the task and replicate of `context`.

    Writes one record per selection method and mixture at each pool size (setting "K=<size>"),
    one for Instruction (setting "none"), and one `<method>+composition` at the same setting for
    each composed method. Raises pool.PoolError if the task has fewer candidates than a pool
    size."""
    context = context.at_threshold(settings.threshold)
    asked = asked_methods(settings.methods, compose=context.options.compose)
    factory = samplers.Factory(
        context.original,
        context.task.prompt_token_ids,
        composed_part_size=context.generation_batch(samplers.COMPOSED_PART_SIZE),
    )
    session = shared.Session(context, settings, factory, draws.Draws(context, factory))
    summaries: list[run_record.RunSummary] = []
    in_pools = [method for method in asked if not isinstance(method, methods.Instruction)]
    if in_pools and settings.pool_sizes:
        candidates = selection.ordered_candidates(context, max(settings.pool_sizes))
        members: list[selection.Member] = []
        for size in settings.pool_sizes:
            # A larger pool adds candidates to the smaller ones and keeps their selection draws.
            members += selection.draw_selection(session, candidates[len(members) : size])
            for method in in_pools:
                if isinstance(method, methods.Selection):
                    returned = selection.select(session, members[:size], method)
                else:
                    returned = selection.mix(session, members[:size], method)
                summaries.append(returned.recorded.summary)
                if asked[method]:
                    summaries.append(compose(session, returned).summary)
    if methods.Instruction() in asked:
        returned = instruct(session)
        summaries.append(returned.recorded.summary)
        if asked[methods.Instruction()]:
            summaries.append(compose_instruction(session, returned).summary)
    return summaries


def asked_methods(names: Sequence[str], *, compose: bool) -> dict[methods.Method, bool]:
    """The methods `names` ask for, in order, each with whether its composition is asked for. A
    "+composition" name asks for both, or for nothing without `compose`."""
    asked: dict[methods.Method, bool] = {}
    for name in names:
        method, composed = methods.parse(name)
        if composed and not compose:
            continue
        asked[method] = asked.get(method, False) or composed
    return asked


def instruction_text(task: data.Task) -> str:
    """The Instruction text of the task's behavior, from the package or, for a generalization
    task, from `generalization.instructions`."""
    if isinstance(task, generalization_tasks.GeneralizationTask):
        return generalization_instructions.instruction_for(task)
    return instructions.instruction_for(task.behavior)


def instruct(session: shared.Session) -> shared.Returned:
    """Instruction: the original model under the behavior's fixed system prompt. No selection
    and no threshold test, so it costs nothing."""
    context = session.context
    method = methods.Instruction().name
    setting = run_record.setting_label()
    text = instruction_text(context.task)
    spec = samplers.Spec((tuple(context.original.render_prompt(context.task.prompt, system=text)),))
    spend = ledger.Ledger()
    recorded = context.evaluate_and_record(
        session.factory.build(spec, name=method),
        method=method,
        setting=setting,
        settings=session.settings,
        spend=spend,
        submitted={"system_prompt": text},
    )
    return shared.Returned(method, setting, spec, "no_selection", spend, recorded)


def compose_instruction(session: shared.Session, plain: shared.Returned) -> run_record.Recorded:
    """Instruction + composition. Instruction has no threshold test of its own, so this first
    runs the rank-1 test at the run's threshold, and composes only if it passes.

    Instruction is the same sampler in every replicate, so its composition selection draws come
    from one stream shared by all replicates; its threshold tests are drawn per replicate."""
    assert plain.spec is not None, "Instruction always has a sampler"
    test = rank_table.threshold_test(1, threshold=selection.run_rule(session).threshold)
    rows = session.draws.threshold_test(plain.spec, test.draws)
    hits = draws.count_hits(rows)
    passed = hits >= test.required_hits
    spend = ledger.Ledger.resume(plain.spend.totals())
    charges.apply(spend, charges.threshold_test(test.draws, note="threshold test of Instruction"))
    outcome = {"draws": test.draws, "required_hits": test.required_hits, "hits": hits}
    tested = dataclasses.replace(plain, status="passed" if passed else "best_observed", spend=spend)
    return compose(
        session,
        tested,
        selection_shared_by_replicates=True,
        details={"threshold_test": {**outcome, "passed": passed}},
        draws={
            "threshold_tests": [{"candidate_id": plain.method, "rank": 1, **row} for row in rows]
        },
    )


def compose(
    session: shared.Session,
    plain: shared.Returned,
    *,
    selection_shared_by_replicates: bool = False,
    details: dict[str, Any] | None = None,
    draws: dict[str, list[dict[str, Any]]] | None = None,
) -> run_record.Recorded:
    """Write the record of `<method>+composition` (`composition.compose`) for the sampler `plain`
    returned. A mixture is composed with one setting for all its components. `details` and
    `draws` are added to the record."""
    base, limit = plain.spec, session.factory.response_limit
    components = 1 if base is None else len(base.prompts)

    def composed(setting: composition.Setting) -> samplers.Spec:
        assert base is not None, "a method without a sampler is not composed"
        return base.composed(setting)

    def selection_draws(setting: composition.Setting) -> list[dict[str, Any]]:
        assert base is not None, "a method without a sampler is not composed"
        return session.draws.composition_selection(
            base, setting, shared_by_replicates=selection_shared_by_replicates
        ).rows()

    return composition.compose(
        session.context,
        plain.recorded,
        settings=session.settings,
        spend=plain.spend,
        plain_status=plain.status,
        selection_draws=selection_draws,
        threshold_test=lambda setting, count: session.draws.threshold_test(
            composed(setting), count
        ),
        sampler=lambda setting: session.factory.build(
            composed(setting), name=plain.method + methods.COMPOSITION
        ),
        sampler_id=lambda setting: composed(setting).sampler_id,
        charge=lambda outcome: composition.composition_charges(
            outcome, components=components, response_limit=limit
        ),
        details=details,
        draws=draws,
    )
