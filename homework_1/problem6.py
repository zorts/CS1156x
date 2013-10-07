knownVectors = [((0, 0, 0), 0),
                ((0, 0, 1), 1),
                ((0, 1, 0), 1),
                ((0, 1, 1), 0),
                ((1, 0, 0), 1)]
remaining = ((1, 0, 1),
             (1, 1, 0),
             (1, 1, 1))
             
def xor(a,b):
    return a ^ b
                 
hypotheses = (('evaluateAsOne',    lambda v: 1),
              ('evaluateAsZero',   lambda v: 0),
              ('evaluateAsXOR',    lambda v: reduce(xor, v)),
              ('evaluateAsNotXOR', lambda v: 1 ^ reduce(xor, v))
              )

def evaluateForRemaining(g):
    return [(x, g(x)) for x in remaining]
    
def evaluate(g):
    return knownVectors + evaluateForRemaining(g)

def numberMatching(gVector, tVector):
    result = 0
    for i in range(0,len(gVector)):
        if (gVector[i] == tVector[i]):
            result = result + 1
    return result

def score(g):
    gVector = [g(x) for x in remaining]
    counts = [0,0,0,0]
    for a in (0,1):
        for b in (0,1):
            for c in (0,1):
                tVector = (a,b,c)
                matchingPoints = numberMatching(gVector, tVector)
                counts[matchingPoints] += 1
    result = 0
    for i in range(0,len(counts)):
        result += i * counts[i]
    return result

for name, g in hypotheses:
    print('\n{name}: score {score}'.format(name=name, score=score(g)))
    for x in evaluate(g):
        print(x)
