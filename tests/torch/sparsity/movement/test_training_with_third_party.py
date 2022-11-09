from pathlib import Path
import pytest
import os

from tests.common.helpers import PROJECT_ROOT
from tests.common.helpers import TEST_ROOT
from tests.torch.helpers import Command
from tests.torch.test_sanity_third_party import create_command_line


class TestMovementWithTransformers:
    @pytest.fixture(autouse=True)
    def setup(self, temp_folder):
        self.VENV_PATH = str(temp_folder["venv"])
        self.VENV_ACTIVATE = str(". {}/bin/activate".format(self.VENV_PATH))
        self.PYTHON_EXECUTABLE = str("{}/bin/python".format(self.VENV_PATH))
        self.TRANSFORMERS_REPO_PATH = str(os.path.join(temp_folder["repo"], "transformers"))
        self.CUDA_VISIBLE_STRING = "export CUDA_VISIBLE_DEVICES=0;"
        self.PATH_TO_PATCH = str(os.path.join(PROJECT_ROOT, "third_party_integration", "huggingface_transformers",
                                              "0001-Modifications-for-NNCF-usage.patch"))

    @pytest.mark.dependency(
        depends=['test_sanity_third_party.py::TestTransformers::install_trans'], # TODO:(yujie): how to do cross-file dependency?
        scope="session"
    )
    def test_movement_glue_train(self, temp_folder):
        com_line = "examples/pytorch/text-classification/run_glue.py --model_name_or_path " \
                   "google/bert_uncased_L-2_H-128_A-2 --task_name mrpc --do_train " \
                   " --per_gpu_train_batch_size 4 --learning_rate 1e-4 --num_train_epochs 0.1 --max_seq_length 128 " \
                   " --output_dir {output_dir} --save_steps 200 --nncf_config" \
                   " {nncf_config} " \
            .format(output_dir=os.path.join(temp_folder["models"], "bert-tiny-uncased-glue-mrpc-movement"),
                    nncf_config=Path(TEST_ROOT, 'torch', 'sparsity', 'movement', 'examples', 'bert_tiny_uncased_mrpc_movement.json'))
        runner = Command(create_command_line(com_line, self.VENV_ACTIVATE, self.PYTHON_EXECUTABLE,
                                             self.CUDA_VISIBLE_STRING), self.TRANSFORMERS_REPO_PATH)
        runner.run()
        assert os.path.exists(os.path.join(temp_folder["models"], "bert-tiny-uncased-glue-mrpc-movement", "pytorch_model.bin"))
    # Plan to add scripts for wav2vec & swin.