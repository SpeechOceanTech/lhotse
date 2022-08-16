from typing import Sequence

import click

from lhotse.bin.modes import prepare
from lhotse.recipes.so_kaldi import prepare_kalditype_speech
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-s",
    "--subset-name",
    type=str,
    default="dev",
    help="How many threads to use (can give good speed-ups with slow disks).",
)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
def so_kaldi(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    subset_name: str,
    num_jobs: int,
):
    """
    The SpeechOcean Kaldi Type corpus preparation.
    """
    prepare_kalditype_speech(
        corpus_dir,
        output_dir=output_dir,
        subset_name=subset_name,
        num_jobs=num_jobs,
    )
