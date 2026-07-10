from testloop.runner import run_tests

src = open("example_target.py").read()
tests = "import target\ndef test_add():\n    assert target.add(2, 3) == 5\n"

r = run_tests(src, tests)
print("passed:", r.passed, "| failed:", r.failed, "| errors:", r.errors)
print("collected:", r.collected, "| coverage:", r.coverage, "| timed_out:", r.timed_out)
print("=========== RAW PYTEST OUTPUT ===========")
print(r.output)
print("=========================================")