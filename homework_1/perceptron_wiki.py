from operator import add
import argparse
import random

class Perceptron(object):
    """Implement a simple Perceptron learning neuron."""

    weight = []

    def __init__(self, dimensions=2):
        """Create a perceptron with the specified number of input dimensions.
        """
        assert dimensions > 0
        for d in range(0,dimensions+1):
            self.weight.append(0)


    def __evaluate(self, point):
        """Return the result of evaluating the perceptron for the specified point.
        The result is NOT normalized to -1 or 1.
        """
        assert len(point) == len(self.weight)-1
        result = self.weight[0]
        for i in range(0,len(point)):
            result += self.weight[i+1] * point[i]
        return result
    

    def evaluate(self, point):
        """Return the result of evaluating the perceptron for the specified point.
        """
        result = self.__evaluate(point)
        return -1 if result < 0 else 1
    

    def train(self, exemplars, learningRate, iterationLimit, errorThreshold):
        """ Train the perceptron on a set of exemplars with the specified learning rate, iteration limit
        and error threshold and return information about the success or failure of the training.
    
        Keyword arguments: 
        exemplars: a set of known points to be used to train the perceptron. Each point consists of a 
                   pair of values: a tuple providing the dimensional values for the point, and the result
                   for the point.
        learningRate: a number between 0 and 1 that controls the slew rate of the weights.
        iterationLimit: the maximum number of passes over the exemplars to be made before abandoning training.
        errorThreshold: the largest permissible difference between the actual and computed results for the
                        points in the training set.
        Return: the success or failure of the training (True/False) and the number of iterations executed.
        """
        assert 0 < learningRate
        assert 1 >= learningRate
        assert len(exemplars) > 1

        for iteration in range(0, iterationLimit):
            for point, desiredResult in exemplars:
                currentResult = self.__evaluate(point)
                self.__adjustWeights(point, desiredResult, currentResult, learningRate)
            if (self. __trainingComplete(exemplars, errorThreshold)):
                return True, iteration
        
        return False, iterationLimit

    def __adjustWeights(self, point, desiredResult, currentResult, learningRate):
        for i in range(0, len(point)):
            print("adjust: i={i}, weight={weight}, desired={desired}, current={current}".format(i=i, weight=self.weight[i+1], desired=desiredResult, current=currentResult))
            self.weight[i+1] += learningRate * (desiredResult - currentResult) * point[i]

    def __trainingComplete(self, exemplars, errorThreshold):
        error = 0;
        for vector, expected in exemplars:
            actual = self.__evaluate(vector)
            diff = abs(actual - expected);
            error += diff
            
#        error = (1 / len(exemplars)) * reduce(add, [x[1] - self.__evaluate(x[0]) for x in exemplars])
        print("error={error}, threshold={threshold}".format(error=error, threshold=errorThreshold))
        return error <= errorThreshold

def correctValueForVector(x, y, slope, intercept):
    if (y > (slope*x + intercept)):
        value = -1
    else:
        value = 1
    return value

def generatePoint(slope, intercept, rng):
    x = rng.uniform(-1, 1)
    y = rng.uniform(-1, 1)
    return ((x,y), correctValueForVector(x,y, slope, intercept))

def generatePoints(slope, intercept, numberOfPoints, seed):
    rng = random.Random()
    rng.seed(seed)
    return [generatePoint(slope, intercept, rng) for i in range(0, numberOfPoints)]

def InterceptType(string):
    value = float(string)
    if (value <= -1 or value >= 1):
        msg = "The Y intercept must be between -1 and 1; %r is invalid." % value
        raise argparse.ArgumentTypeError(msg)
    return value

def ErrorThreshold(string):
    value = float(string)
    if (value <= 0 or value >= 1):
        msg = "The error threshold must be between 0 and 1; %r is invalid." % value
        raise argparse.ArgumentTypeError(msg)
    return value

def LearningRate(string):
    value = float(string)
    if (value <= 0 or value >= 1):
        msg = "The learning rate must be between 0 and 1; %r is invalid." % value
        raise argparse.ArgumentTypeError(msg)
    return value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slope", "-s", help="slope of the plane separator", default=1, type=float)
    parser.add_argument("--intercept", "-m", help="Y intercept of the plane separator, [-1..1]", default=0, type=InterceptType)
    parser.add_argument("--numberOfPoints", "-n", help="number of points to be generated in the training set", default=10, type=int)
    parser.add_argument("--learningRate", "-r", help="learning rate", default=.1, type=LearningRate)
    parser.add_argument("--errorThreshold", "-e", help="error threshold, [0..1]", default=.1, type=ErrorThreshold)
    parser.add_argument("--iterationLimit", "-i", help="maximum number of training iterations", default=1000, type=int)
    parser.add_argument("--seed", "-x", help="seed for random number generator", default=None)
    parser.add_argument("--testPoints", "-t", help="number of test points", default=10, type=int)
    args = parser.parse_args()
    
    print(args)
    trainingSet = generatePoints(args.slope, args.intercept, args.numberOfPoints, args.seed)
    p = Perceptron()
    trainingSucceeded, iterations = p.train(trainingSet, args.learningRate, args.iterationLimit, args.errorThreshold)
    print("training {success} after {iterations} iterations."
          .format(success="succeeded" if trainingSucceeded else "failed",
                  iterations=iterations)
          )
    if (trainingSucceeded):
        testSet = generatePoints(args.slope, args.intercept, args.testPoints, args.seed)
        for vector, expected in testSet:
            result = p.evaluate(vector)
            if (abs(result - expected) > args.errorThreshold):
                print("Error for ({x}, {y}): expected {expected}, got {result}."
                      .format(x=vector[0], y=vector[1], expected=expected, result = result))
        

if __name__ == "__main__":
    main()
