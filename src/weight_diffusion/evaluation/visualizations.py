import seaborn as sns
from weight_diffusion.evaluation.constants import PA_KEY, TARGETS_KEY, TEST_ACC_KEY


def plot_prompt_alignment(logging_dict):
    prompt_alignments = [
        (v[PA_KEY], v[TARGETS_KEY][TEST_ACC_KEY]) for _, v in logging_dict.items()
    ]
    ax = sns.barplot(
        data=prompt_alignments,
    )
    ax.set_xticklabels([x[1] for x in prompt_alignments])
    ax.set_ylabel("Prompt Alignment (RMSE)")
    ax.set_xlabel("Prompted Test accuracy")
    ax.set_title("Prompt alignment across different prompts")


def plot_desired_vs_observed(
    logging_dict, metric_key: str = "test_acc", epoch: int = 0
):
    desired = [v[TARGETS_KEY][metric_key] for k, v in logging_dict.items()]
    observed = [v[f"epoch_{epoch}"][metric_key] for _, v in logging_dict.items()]
    ax = sns.lineplot(
        x=desired,
        y=observed,
    )
    ax.set_title(f"Desired vs. Observed {metric_key}")
    ax.set_xlabel(f"Desired {metric_key}")
    ax.set_ylabel(f"Observed {metric_key}")
