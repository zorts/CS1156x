from operator import add
import argparse
import random

class Perceptron(object):
    """Implement a simple Perceptron learning neuron."""

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
        errorThreshold: the initial value for the threshold (becomes w0)
        Return: the success or failure of the training (True/False) and the number of iterations executed.
        """
        assert 0 < learningRate
        assert 1 >= learningRate
        assert len(exemplars) > 1
        self.weight[0] = errorThreshold

        for iteration in range(0, iterationLimit):
            misclassified = 0
            # TBD: take control of the seed for the RNG so that results are reproducable. To
            #      get reproducable results without doing that work, remove the shuffle, but
            #      the training performance will be inferior.
            random.shuffle(exemplars)
            for point, desiredResult in exemplars:
                currentResult = self.evaluate(point)
                if (desiredResult != currentResult):
                    self.__adjustWeights(point, desiredResult, currentResult, learningRate)
                    misclassified += 1
                    break
            if (misclassified == 0):
                return True, iteration+1
        
        return False, iterationLimit


    @staticmethod
    def generatePoints(slope, intercept, numberOfPoints, seed):
        """Generate 2D test data for the Perceptron, within a plane bounded by [-1,-1] and [1,1].

        The plane will be divided into two regions by a line defined by the specified slope and Y intercept. 
        Points are generated with random x-y coordinates and defined as being in one of the regions by (-1, +1). 
        The resulting points look like this:  ((x, y), region)
        """
        rng = random.Random()
        rng.seed(seed)
        return [Perceptron.__generatePoint(slope, intercept, rng) for i in range(0, numberOfPoints)]

    ## Private functions

    def __init__(self, dimensions=2):
        """Create a perceptron with the specified number of input dimensions.
        """
        assert dimensions > 0
        self.weight = [0 for i in range(0,dimensions+1)]

    def __evaluate(self, point):
        """Return the result of evaluating the perceptron for the specified point.
        The result is NOT normalized to -1 or 1.
        """
        x = (1,) + point
        result = 0
        for i in range(0,len(x)):
            result += self.weight[i] * x[i]
        return result
   
    def __adjustWeights(self, point, desiredResult, currentResult, learningRate):
        self.weight[0] += desiredResult*learningRate
        for i in range(1, len(self.weight)):
            self.weight[i] += desiredResult*learningRate*point[i-1]


    @staticmethod
    def __generatePoint(slope, intercept, rng):
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        return ((x,y), Perceptron.__correctValueForVector(x,y, slope, intercept))


    @staticmethod
    def __correctValueForVector(x, y, slope, intercept):
        return (-1 if (y > (slope*x + intercept)) else 1)

#####################################3

def InterceptType(string):
    value = float(string)
    if (value <= -1 or value >= 1):
        msg = "The Y intercept must be between -1 and 1; %r is invalid." % value
        raise argparse.ArgumentTypeError(msg)
    return value

def ErrorThreshold(string):
    value = float(string)
    if (value < 0 or value > 1):
        msg = "The error threshold must be between 0 and 1; %r is invalid." % value
        raise argparse.ArgumentTypeError(msg)
    return value

def LearningRate(string):
    value = float(string)
    if (value <= 0 or value > 1):
        msg = "The learning rate must be between 0 and 1; %r is invalid." % value
        raise argparse.ArgumentTypeError(msg)
    return value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slope", "-s", help="slope of the plane separator", default=1, type=float)
    parser.add_argument("--intercept", "-m", help="Y intercept of the plane separator, [-1..1]", default=0, type=InterceptType)
    parser.add_argument("--numberOfPoints", "-n", help="number of points to be generated in the training set", default=10, type=int)
    parser.add_argument("--learningRate", "-r", help="learning rate", default=1, type=LearningRate)
    parser.add_argument("--errorThreshold", "-e", help="error threshold, [0..1]", default=0, type=ErrorThreshold)
    parser.add_argument("--iterationLimit", "-i", help="maximum number of training iterations", default=1000, type=int)
    parser.add_argument("--seed", "-x", help="seed for random number generator", default=None)
    parser.add_argument("--testPoints", "-t", help="number of test points", default=10, type=int)
    args = parser.parse_args()
    
    print(args)
    trainingSet = Perceptron.generatePoints(args.slope, args.intercept, args.numberOfPoints, args.seed)
    p = Perceptron()
    trainingSucceeded, iterations = p.train(trainingSet, args.learningRate, args.iterationLimit, args.errorThreshold)
    print("training {success} after {iterations} iterations."
          .format(success="succeeded" if trainingSucceeded else "failed",
                  iterations=iterations)
          )
    if (trainingSucceeded):
        testSet = Perceptron.generatePoints(args.slope, args.intercept, args.testPoints, str(args.seed) + "foo")
        for vector, expected in testSet:
            result = p.evaluate(vector)
            if (result != expected):
                print("Error for ({x}, {y}): expected {expected}, got {result}."
                      .format(x=vector[0], y=vector[1], expected=expected, result = result))

if __name__ == "__main__":
    main()
