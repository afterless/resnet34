import tempfile
import os
import time
import torch as t
import transformers
import joblib
import requests
import logging
import http
from functools import wraps
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from typing import Optional

mem = joblib.Memory(tempfile.gettempdir() + "/joblib_cache")


@mem.cache
def load_pretrained_gpt() -> GPT2LMHeadModel:
    """Load the HuggingFace GPT-2.

    On first use this downloads about 500MB from the Internet.
    Later uses should hit the cache and take under 1s to load.
    """
    return transformers.AutoModelForCausalLM.from_pretrained("gpt2")


@mem.cache
def load_pretrained_bert() -> BertForMaskedLM:
    """Load the HuggingFace BERT.

    Supresses the spurious warning about some weights not being used.
    """
    logger = logging.getLogger("transformers.modeling_utils")
    was_disabled = logger.disabled
    logger.disabled = True
    bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")
    logger.disabled = was_disabled
    return bert


def assert_all_equal(actual: t.Tensor, expected: t.Tensor):
    mask = actual == expected
    if not mask.all():
        bad = mask.nonzero()
        msg = f"Did not match at {len(bad)} indexes: {bad[:10]}{'...' if len(bad) > 10 else ''}"
        raise AssertionError(f"{msg}\nActual:\n{actual}\nExpected:\n{expected}")


def test_is_equal(actual: t.Tensor, expected: t.Tensor, test_name: str):
    try:
        run_and_report(assert_all_equal, test_name, actual, expected)
    except AssertionError as e:
        print(f"Test failed: {test_name}")
        raise e


def assert_shape_equal(actual: t.Tensor, expected: t.Tensor):
    if actual.shape != expected.shape:
        raise AssertionError(f"expected shape={expected.shape}, got {actual.shape}")


def allclose(actual: t.Tensor, expected: t.Tensor, rtol=1e-4, atol: Optional[float] = None) -> None:
    if (rtol is None) == (atol is None):
        raise Exception("This version of allclose expects exactly one of rtol and atol")

    assert_shape_equal(actual, expected)

    left = (actual - expected).abs()
    if rtol is not None:
        right = rtol * expected.abs()
        pct_wrong = int(100 * (left > right).float().mean())
    elif atol is not None:
        pct_wrong = int(100 * (left > atol).float().mean())
    else:
        raise Exception("Bad arguments")

    if pct_wrong > 0:
        print(f"Test failed. Max absolute deviation: {left.max()}")
        print(f"Actual:\n{actual}\nExpected:\n{expected}")
        raise AssertionError(f"allclose failed with {pct_wrong} percent of entries outside tolerance")


def allclose_atol(actual: t.Tensor, expected: t.Tensor, atol: float) -> None:
    assert_shape_equal(actual, expected)
    left = (actual - expected).abs()
    pct_wrong = int(100 * (left > atol).float().mean())
    if pct_wrong > 0:
        print(f"Test failed. Max absolute deviation: {left.max()}")
        print(f"Actual:\n{actual}\nExpected:\n{expected}")
        raise AssertionError(f"allclose failed with {pct_wrong} percent of entries outside tolerance")


def allclose_scalar(actual: float, expected: float, rtol=1e-4, atol: Optional[float] = None) -> None:
    if (rtol is None) == (atol is None):
        raise Exception("This version of allclose expects exactly one of rtol and atol")
    left = abs(actual - expected)
    if rtol is not None:
        right = rtol * abs(expected)
        wrong = left > right
    elif atol is not None:
        wrong = left > atol
    else:
        raise Exception("Bad arguments")

    if wrong:
        print(f"Test failed. Absolute deviation: {left}")
        print(f"Actual:\n{actual}\nExpected:\n{expected}")


def report_success(testname):
    """POST to the server indicating success at the given test.

    Used to help the TAs know how long each section takes to complete.
    """
    server = os.environ.get("MLAB_SERVER")
    email = os.environ.get("MLAB_EMAIL")
    if server:
        if email:
            r = requests.post(
                server + "/api/report_success",
                json=dict(email=email, testname=testname),
            )
            if r.status_code != http.HTTPStatus.NO_CONTENT:
                raise ValueError(f"Got status code from server: {r.status_code}")
        else:
            raise ValueError(f"Server set to {server} but no MLAB_EMAIL set!")
    else:
        if email:
            raise ValueError(f"Email set to {email} but no MLAB_SERVER set!")
        else:
            return  # local dev, do nothing


# Map from qualified name "test_w2d3.test_unidirectional_attn" to whether this test was passed in the current interpreter session
# Note this can get clobbered during autoreload
TEST_FN_PASSED = {}


def report(test_func):
    name = f"{test_func.__module__}.{test_func.__name__}"
    # This can happen when using autoreload, so don't complain about it.
    # if name in TEST_FN_PASSED:
    #     raise KeyError(f"Already registered: {name}")
    TEST_FN_PASSED[name] = False

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        return run_and_report(test_func, name, *args, **kwargs)

    return wrapper


def run_and_report(test_func, name, *test_func_args, **test_func_kwargs):
    start = time.time()
    out = test_func(*test_func_args, **test_func_kwargs)
    elapsed = time.time() - start
    print(f"{name} passed in {elapsed:.2f}s.")
    if not TEST_FN_PASSED.get(name):
        report_success(name)
        TEST_FN_PASSED[name] = True
    return out


def remove_hooks(module: t.nn.Module):
    """Remove all hooks from module.

    Use module.apply(remove_hooks) to do this recursively.
    """
    module._backward_hooks.clear()
    module._forward_hooks.clear()
    module._forward_pre_hooks.clear()
