from agent_quality.agent import ModelRateLimiter, model_limiter


def test_model_limiter_is_shared_for_same_model():
    assert model_limiter("test-model", 12) is model_limiter("test-model", 12)


def test_rate_limiter_interval():
    limiter = ModelRateLimiter(12)
    assert limiter.interval == 5.0
