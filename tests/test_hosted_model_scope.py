from __future__ import annotations

import pytest

from do_benchmark.core import (
    DIGITALOCEAN_HOSTED_MODEL_IDS,
    choose_models,
    require_digitalocean_hosted_models,
)
from do_benchmark.direct_aimd import default_model_ids as aimd_default_model_ids
from do_benchmark.direct_capability import (
    default_model_ids as capability_default_model_ids,
)
from do_benchmark.direct_closure import default_model_ids as closure_default_model_ids
from do_benchmark.direct_completion import (
    default_model_ids as completion_default_model_ids,
)
from do_benchmark.direct_context import default_model_ids as context_default_model_ids
from do_benchmark.direct_soak import default_model_ids as soak_default_model_ids


def test_all_spend_bearing_defaults_are_digitalocean_hosted_only() -> None:
    defaults = (
        aimd_default_model_ids(),
        soak_default_model_ids(),
        capability_default_model_ids(),
        context_default_model_ids(),
        completion_default_model_ids(),
        closure_default_model_ids(),
    )
    assert all(model_ids == DIGITALOCEAN_HOSTED_MODEL_IDS for model_ids in defaults)
    assert "arcee-trinity-large-thinking" not in DIGITALOCEAN_HOSTED_MODEL_IDS
    assert "deepseek-v4-flash-0731" in DIGITALOCEAN_HOSTED_MODEL_IDS


def test_partner_model_cannot_be_selected_explicitly() -> None:
    with pytest.raises(ValueError, match="non-DigitalOcean-hosted"):
        require_digitalocean_hosted_models(("arcee-trinity-large-thinking",))
    with pytest.raises(ValueError, match="non-DigitalOcean-hosted"):
        choose_models(("arcee-trinity-large-thinking",))
