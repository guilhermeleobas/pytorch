
for f in $(ls test/dynamo/cpython/3.13/test_*.py); do
    echo "testing $f"
    PYTORCH_TEST_WITH_DYNAMO=1 python $f -v &> $f.log
done