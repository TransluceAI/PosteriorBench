# Bundled dataset tables

`tasks/tasks.parquet` and `evaluation/evaluation.parquet` are the task table and the evaluation
file of the PosteriorBench dataset, version v1.0.0. They are part of the dataset, not of the
package's code: they are released under the Open Data Commons Attribution License 1.0
([LICENSE](LICENSE)), the license of WildChat-1M, whose prompts the dataset uses. The code is
released under the MIT license.

The evaluation file holds responses of a language model, which were not reviewed by people. They
include offensive language and racial slurs, responses that discuss or encourage suicide and
self-harm, and false statements.
