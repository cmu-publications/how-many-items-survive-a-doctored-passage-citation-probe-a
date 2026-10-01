from data import AsqaAlceDataset, item_rng, pool_for_seed, scan_order
from testing_fakes import make_item


def _ds(n_dev=10, n_conf=6):
    dev = [make_item(f"d{i}", "Paris") for i in range(n_dev)]
    conf = [make_item(f"c{i}", "Berlin") for i in range(n_conf)]
    return {"val": AsqaAlceDataset(dev, role="development_pool"),
            "test": AsqaAlceDataset(conf, role="confirmation_pool")}


def _ids(pool):
    return sorted(str(it["sample_id"]) for it in pool)


def test_every_dev_seed_reads_whole_dev_pool():
    ds = _ds()
    for s in (0, 1, 2, 3, 4):
        assert _ids(pool_for_seed(s, [0, 1, 2, 3, 4], ds)) == _ids(ds["val"].items)


def test_non_plan_seed_reads_whole_confirmation_pool():
    ds = _ds()
    for s in (5, 11, 123):
        pool = pool_for_seed(s, [0, 1, 2, 3, 4], ds)
        assert _ids(pool) == _ids(ds["test"].items)
        assert not set(_ids(pool)) & set(_ids(ds["val"].items))


def test_pool_choice_ignores_other_seeds():
    ds = _ds()
    assert _ids(pool_for_seed(2, [2], ds)) == _ids(pool_for_seed(2, [4, 3, 2, 1, 0], ds))
    assert _ids(pool_for_seed(9, [0], ds)) == _ids(pool_for_seed(9, [0, 1, 2], ds))


def test_item_rng_order_independent():
    a = item_rng(3, "q1", "target").random(4)
    _ = item_rng(3, "q2", "target").random(4)
    b = item_rng(3, "q1", "target").random(4)
    assert list(a) == list(b)
    assert list(a) != list(item_rng(3, "q1", "foil").random(4))
    assert list(a) != list(item_rng(4, "q1", "target").random(4))


def test_scan_order_deterministic():
    ds = _ds()
    pool = list(ds["val"].items)
    o1 = scan_order(pool, 1)
    assert o1 == scan_order(list(reversed(pool)), 1)
    assert sorted(o1) == _ids(pool)
    assert o1 != scan_order(pool, 2) or len(pool) < 3