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

## II.example
    priklad s fixed point: to neznam nebo jsme zapomnel podivat se na to
    ```scala
    val tolerance = 0.0001
    def isCloseEnough(x: Double, y: Double) = abs((x - y) / x) / x < tolerance
    def fixedPoint(f: Double => Double)(firstGuess: Double) = {
        def iterate(guess: Double): Double = {
            val next = f(guess)
                //println(next)
                if (isCloseEnough(guess, next)) next
                else iterate(next)
        }
        iterate(firstGuess)
    }
    def sqrt(x: Double) = fixedPoint(y => x / y)(1.0)
    def sqrt(x: Double) = fixedPoint(y => (y + x / y) / 2)(1.0)
    def averageDamp(f: Double => Double)(x: Double) = (x + f(x)) / 2
    def sqrt(x: Double) = fixedPoint(averageDamp(y => x/y))(1.0)
```
### EBNF  Extended Backus-Naur form
|,[...] an option (0 or 1),{...} a repetition (0 or more).

# III Functions And Data (Classes)
```scala
    class Rational(x: Int, y: Int) {
    require(y > 0, ”denominator must be positive”) ;;IllegalArgument
        def this(x: Int) = this(x, 1)  //JINY NEZ DEFUALTNI KONSTRUKTOR!!!!
        private def gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)
        private val g = gcd(x, y) //!!private
        def numer = x /g
        def denom = y/g
        def add(r: Rational) = new Rational(numer * r.denom + r.numer * denom, denom * r.denom) //method
        //def mul(r: Rational) = ...
        def *(r: Rational) = ...//je mozne pouzit i specialni znaky
        def less(that: Rational) = this.numer * that.denom < that.numer * denom
        def max(that: Rational)  = if (this.less(that)) that else this
        //assert(x >= 0) //asert se oiuziva uvnitr funkci
        override def toString = numer + ”/” + denom        
    }
    x.add(y).mul(z)    
```
### Evaluation
 new Rational(p1,p2,..).method(q1,q2,q3) 
 3 kroky
    * parametry pri vytvareni
    * parametry metody
    * nahrazeni this
###Operatory    
    je mozne psat
    ```scala
    r.max(s) 
    r max s //je to to same
    ```
    Precedence
    > letters,|,^,&,<>,=!,:,+-,*/%,other special chars
    
    unary_- rozdil od binary - jako operand infix notation

## IIIb hierarchie/abstract/spoluprace mezi classy



TEDNE: asi by se to hodilo pro strukturu funsetu, ale zatim staci jiny namespace
clojure: type vs record vs map
deftype mutable defrecord inmutablemaplike
http://clojure.org/reference/datatypes ...why 2?
--
defprotocol
extends, extend-type,extend-protocol i existujici typy String/nil
reify
