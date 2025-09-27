import os

import pytest

from timecopilot.models.foundation.timesfm import _TimesFMV1, _TimesFMV2_p5

# Define parameters for each model version.
# Each tuple contains: (class_to_test, path_to_underlying_lib_class)
MODEL_PARAMS = [
    (_TimesFMV1, "timecopilot.models.foundation.timesfm.timesfm_v1.TimesFm"),
    (
        _TimesFMV2_p5,
        "timecopilot.models.foundation.timesfm.timesfm.TimesFM_2p5_200M_torch",
    ),
]


@pytest.mark.parametrize("model_class, lib_class_path", MODEL_PARAMS)
def test_model_loads_from_local_path(mocker, model_class, lib_class_path):
    """Tests loading from a local path."""
    # --- Setup ---
    module_path = "timecopilot.models.foundation.timesfm"
    mock_os_exists = mocker.patch(f"{module_path}.os.path.exists", return_value=True)
    mock_lib_class = mocker.patch(lib_class_path)
    local_path = "/fake/local/path"

    # --- Action ---
    model_instance = model_class(
        repo_id=local_path, context_length=64, batch_size=32, alias="test"
    )
    with model_instance._get_predictor(prediction_length=12):
        pass

    # --- Assert ---
    mock_os_exists.assert_called_once_with(local_path)

    if model_class is _TimesFMV1:
        mock_lib_class.assert_called_once()
    elif model_class is _TimesFMV2_p5:
        expected_call = {"path": os.path.join(local_path, "model.safetensors")}
        mock_lib_class.return_value.load_checkpoint.assert_called_once_with(
            **expected_call
        )


@pytest.mark.parametrize("model_class, lib_class_path", MODEL_PARAMS)
def test_model_loads_from_hf_repo(mocker, model_class, lib_class_path):
    """Tests loading from a Hugging Face repo."""
    # --- Setup ---
    module_path = "timecopilot.models.foundation.timesfm"
    mock_os_exists = mocker.patch(f"{module_path}.os.path.exists", return_value=False)
    mock_lib_class = mocker.patch(lib_class_path)
    repo_id = "google/fake-repo-id"

    # --- Action ---
    model_instance = model_class(
        repo_id=repo_id, context_length=64, batch_size=32, alias="test"
    )
    with model_instance._get_predictor(prediction_length=12):
        pass

    # --- Assert ---
    mock_os_exists.assert_called_once_with(repo_id)

    if model_class is _TimesFMV1:
        mock_lib_class.assert_called_once()
    elif model_class is _TimesFMV2_p5:
        expected_call = {"hf_repo_id": repo_id}
        mock_lib_class.return_value.load_checkpoint.assert_called_once_with(
            **expected_call
        )


@pytest.mark.parametrize("model_class, lib_class_path", MODEL_PARAMS)
def test_model_raises_ioerror_on_failed_load(mocker, model_class, lib_class_path):
    """Tests that an IOError is raised on a failed load attempt."""
    # --- Setup ---
    module_path = "timecopilot.models.foundation.timesfm"
    mocker.patch(f"{module_path}.os.path.exists", return_value=False)

    # Mock the specific failure point for each version
    if model_class is _TimesFMV1:
        mocker.patch(lib_class_path, side_effect=Exception("mocked failure"))
    elif model_class is _TimesFMV2_p5:
        mock_instance = mocker.patch(lib_class_path)
        mock_instance.return_value.load_checkpoint.side_effect = Exception(
            "mocked failure"
        )

    # --- Action & Assert ---
    model_instance = model_class(
        repo_id="bad/path", context_length=64, batch_size=32, alias="test"
    )
    with (
        pytest.raises(IOError, match="Failed to load model"),
        model_instance._get_predictor(prediction_length=12),
    ):
        pass
