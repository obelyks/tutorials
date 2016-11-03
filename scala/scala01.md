# 0: Prednaska
```scala
class Person(val name:String, val age:Int)
val people: Array[Person]
val (minors,adults) = people     partition (_.age < 18 )
val (minors,adults) = people.par partition (_.age < 18 )  //parallel

actor {
    receive {
        case people: Set[Person] => val (minors,adults) = people     partition (_.age < 18 )
        facebook !  minors
        linkedin !  adults
    }
}

for ( i <- 1 to 8) {}
```

# I Substitution model

CBV vyhodnoti parametry fce hned
CBN vyhodnoti parametr funkce teprve jestli je uvnitr fce potrebuje a klidne vickrat to same

cbv terminates=>cbn terminates: naopak to neplati

parametry fce
> * CBV scala default
> * => CBN


```scala
def constOne(x: Int, y: => Int) = 1
constOne(1+2, loop)
constOne(loop, 1+2)
```

definice promennych
> * def CBN //eval at each use
> * val CBV //eval at definition

```scala
def sqrtIter(guess: Double, x: Double): Double =
    if (isGoodEnough(guess, x)) guess
    else sqrtIter(improve(guess, x), x)

def sqrt(x: Double) = {
    def sqrtIter(guess: Double): Double = if (isGoodEnough(guess)) guess else sqrtIter(improve(guess))
    def improve(guess: Double)          = (guess + x / guess) / 2
    def isGoodEnough(guess: Double)     = abs(square(guess) - x) < 0.001
    sqrtIter(1.0)
}

@tailrec
def gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)
```

# II Higher order functions

function type A=>B (Int => Int)
anonymous function = inline

> def f(x :T1, x2: T2...) =  impl
> (x:T1, x2: T2...)       => impl

```scala
def sum(f: Int => Int, a: Int, b: Int): Int = if (a > b) 0 else f(a) + sum(f, a + 1, b)
def sumInts(a: Int, b: Int) = sum(x => x, a, b)
def sumCubes(a: Int, b: Int) = sum(x => x * x * x, a, b)
```

> A => B funkcni typ
> x => x funkce id, 1 parametr, ktery se beze zmeny vraci

## Currying
funkce co vraci fci se 2 parametrama a vracejici 1 cislo
```scala
def sum(f: Int => Int): (Int, Int) => Int = {
    def sumF(a: Int, b: Int): Int = if (a > b) 0 else f(a) + sumF(a + 1, b)
    sumF
}
```
Volamejako sum (cube) (1, 10)  jetotosame jako== (sum (cube)) (1, 10) 

Dalsi zkratka:umoznuje volat i jako sum cube
```scala
def sum(f: Int => Int)(a: Int, b: Int): Int = if (a > b) 0 else f(a) + sum(f)(a + 1, b)
```
### Typ
> (Int => Int) => (Int, Int) => Int
> to je to same jako
> (Int => Int) => ((Int, Int) => Int)
> fuknce parametr typu funkce a vraci funkci (2 int parametry vraci 1 int)

# III example
    priklad s fixed point: to neznam nebo jsme zapomnel podivat se na to

unary_- rozdil od binary - jako operand infix notation
