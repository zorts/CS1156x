from perceptron import Perceptron 
import random
import sys

# TBD: use argparse for this stuff
MAX_ITER = 500
TRAINING_EXEMPLARS = 100
TEST_VECTORS = 1000
TEST_RUNS = 1000
ERROR_THRESHOLD=0

def runOneTest(seed=None):
    sys.stdout.write('.')
    sys.stdout.flush()
    rnd = random.Random()
    rnd.seed(seed)
    slope = rnd.uniform(-3, 3)
    intercept = rnd.uniform(-1,1)
    trainingSet = Perceptron.generatePoints(slope, intercept, TRAINING_EXEMPLARS, str(seed)+"training")
    p = Perceptron()
    trainingSucceeded, iterations = p.train(trainingSet, 1, MAX_ITER, ERROR_THRESHOLD)
    misclassifications = 0
    if (trainingSucceeded):
        testSet = Perceptron.generatePoints(slope, intercept, TEST_VECTORS, str(seed)+"test")
        for vector, expected in testSet:
            result = p.evaluate(vector)
            if (result != expected):
                misclassifications += 1

    return slope, intercept, trainingSucceeded, iterations, misclassifications

def main():
    trainedOK = 0
    minIter, maxIter, totalIter = MAX_ITER, 0, 0.0
    minError, maxError, totalError = TEST_VECTORS, 0, 0.0
    failed = 0
    passed = 0
    print("\nexemplars={n}".format(n=TRAINING_EXEMPLARS))
    for i in range(0,TEST_RUNS):
        slope, intercept, trainingSucceeded, iterations, misclassifications = runOneTest(i)
        if (((i+1) % 100) == 0):
            print(i+1)
        if (trainingSucceeded):
            passed += 1
            totalIter += iterations
            totalError += misclassifications
            minIter = iterations if iterations < minIter else minIter
            maxIter = iterations if iterations > maxIter else maxIter
            minError = misclassifications if misclassifications < minError else minError
            maxError = misclassifications if misclassifications > maxError else maxError
        else:
            failed += 1
    print("")
    print("Iterations/run: min={min}, avg={avg}, max={max}".format(
            min=minIter, avg=totalIter/float(passed), max=maxIter))
    print("Misclassifications/run: min={min}, avg={avg}, max={max}".format(
            min=minError, avg=totalError/float(passed), max=maxError))
    print("Misclassification overall: {pct}%".format(pct=100*totalError/float(passed*TEST_VECTORS)))
    print("Failures: {fail} of {runs} ({pct}%)".format(fail=failed, runs=TEST_RUNS, pct=100*(failed/float(TEST_RUNS))
            ))

if __name__ == "__main__":
    main()
