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
### Operatory    
    je mozne psat
```scala
    r.max(s) 
    r max s //je to to same
```
### Precedence operatoru
    > letters,|,^,&,<>,=!,:,+-,*/%,other special chars
    
    unary_- rozdil od binary - jako operand infix notation

## IIIb hierarchie/abstract/spoluprace mezi classy
abstract classes
```scala
abstract class IntSet {
    def incl(x: Int): IntSet
    def contains(x: Int): Boolean
    def union(other: IntSet): IntSet //TODO asi 3.1 
    }
<del>class</del> object Empty extends IntSet {
    def contains(x: Int): Boolean = false   //nemusi byt override, kdyz implementuje abstraktni metodu
    def incl(x: Int): IntSet = new NonEmpty(x, new Empty, new Empty)
    override def toString="."
    def uninon(other:Intset):IntSet=other
}
class NonEmpty(elem: Int, left: IntSet, right: IntSet) extends IntSet {
    def contains(x: Int): Boolean = if (x < elem) left contains x else if (x > elem) right contains x else true
    def incl(x: Int): IntSet = if (x < elem) new NonEmpty(elem, left incl x, right) else if (x > elem) new NonEmpty(elem, left, right incl x) else this
    override def toString="{"+left +elem +right +"}"
    def uninon(other:Intset):IntSet= ( (left union right) union other) incl elem
}
```

baseclass superclass subclass conforms extends abstract override
class vs object(singleton)
dynamic method dispatch

```scala
object Hello { def main(args: Array[String]) = println(”hello world!”)}
```

packages jako v jave, rozdily:

```scala
import week3.{Rational, Hello} // imports both Rational and Hello
import week3._ // imports everything in package week3
```

automaticky se importuji: java.lang, scala, scala.Predef(singleton)
http://www.scala-lang.org/api/current

### Traits
```scala
trait Planar {
    def height: Int
    def width: Int
    def surface = height * width
}
class Square extends Shape with Planar with Movable ...
```

Any->AnyVal('primitive'),AnyRef(obejct,Iterble,String,...)
Nothing(abnormal termination,elementTypeOfEmptyColection) vs Null(type of null,incompatible withh AnyVal)

# PolymORF
naive implementation of immutable linked list Ćons=List
//List(List(true, false), List(3))

```scala
trait List[T]{ //type parameters, jenom compile time
 def isEmpty:Boolean
 def head:T
 def tail: List[T] }
class Cons[T](val head: T, val tail: List[T]) extends List[T] {    //kdy je tam val tak to nejsou jenom parametry, ale i implementace
    def isEmpty=false}
class Nil[T] extends List[T]{
    def isEmpty=true
    def head= throw new noSuchElementException("head")}
def singleton[T](elem: T) = new Cons[T](elem, new Nil[T])//funkce muzou mit taky type parameter    
singleton[Int](1), singleton[Boolean](true) = singleton(1),singleton(true) //scala compiler to uhadne, nemusi to tam by texplicitne
def nth(n:Int,xs:List[T]):T = 
    if (xs.isEmpty) throw new IndexOfOufBOundException //musi projit cely seznam pri -1 i pri max+1
    else if (n==0) xs.head else nth(n-1,xs.tail)
```

type erasure:type parameter se tyka kompilace //stejne jako java,haskell,ocaml   a narizdil od c++,c#,f#
2typy polymorfismu: subtyping(instance subtypu)  & generics (instance fce nebo classu vytvorene pomoci type parametru)

# IV Typy??
VsechnojeClass: implementace boolean a nat(positivni prirozene cislo)

```scala
package idealized.scala
abstract class Boolean {    //prepsane if cond te then ee na metpdu cond.ifThenElse(te,ee)
    def ifThenElse[T](t: => T, e: => T): T
    def && (x: => Boolean): Boolean = ifThenElse(x, false)
    def || (x: => Boolean): Boolean = ifThenElse(true, x)
    def unary_!: Boolean = ifThenElse(false, true)
    def == (x: Boolean): Boolean = ifThenElse(x, x.unary_!)
    def != (x: Boolean): Boolean = ifThenElse(x.unary_!, x)
    def < (x:Boolean) = ifThenElse(false,x) //exercise!!!
    ...
    }
    object true extends Boolean { def ifThenElse[T](t: => T, e: => T) = t}
    object false extends Boolean { def ifThenElse[T](t: => T, e: => T) = e}
```
a  tedka priklad s cislem "Peano" numbers
```scala
abstract class Nat {
    def isZero: Boolean
    def predecessor: Nat
    def successor: Nat = new Succ(this)
    def + (that: Nat): Nat
    def - (that: Nat): Nat
}
object Zero extends Nat{
    def isZero=true
    def predecessor=throw new Error("negZero")
    def +(that)=that
    def -(that:Nat)= if (that.isZero) this else throw new Eror("neg")
}
class Succ(n: Nat) extends Nat{
    def isZero=false
    def predecessor=n
    def + (that) = new Succ (n+that)
    //def + (that) = new Succ (n-that) //odecita az narazi na nulu a hodi vyjimku!
    def + (that) =  if that.isZero then this else n-that.predecessor
}
Funkce jsou to same, taky objekty, provadi se metoda apply podle poctu parametru
trait Function1[A, B] {
    def apply(x: A): B
}
```
# Ivb Typy Invariant,contra,co ;  LSP(liskov subst princip)
## vztahy mezi generikama a subcassama
TypeBounds
def assertAllPos[S <: IntSet](r: S): S = ... //uper bound   
S <: T means: S is a subtype of T
S >: T means: S is a supertype of T , or T is a subtype of S
[S >: NonEmpty <: IntSet] //lower bound, mixed bounds
covariance... NonEmpty <: IntSet ... List[NonEmpty] <: List[IntSet]
                ovariant because their subtyping relationship varies with the type parameter.
Arrays in Java are covariant:problem...
```java
//JAVA!!!!
NonEmpty[] a = new NonEmpty[]{new NonEmpty(1, Empty, Empty)}
IntSet[] b = a
b[0] = Empty
NonEmpty s = a[0]
```
LSP=If A <: B , then everything one can to do with a value of
type B one should also be able to do with a value of type A 
```scala
funkce: 
type A = IntSet   => NonEmpty
        type B = NonEmpty => IntSet
A<B        A<: B
A: NonEmpty=>nonEmpty(specialcase of IntSet)
Pravidlo A1=>B1  <:  A2 => B2 kdyz je splneno B1<:B2 a zaroven A2 <: A1 (parrametry naopak!)
```

A<: B ... 3 typy covariant C[A] <: C[B]  contravariant C[A] >: C[B] nonvariant C[A],C[B] nejsou subtype
syntax
```scala
    class C[+A] {} //covariant
    class C[-A] {} //contravariant
    class C[A] {}   //nonvariant
```
functions are covariant in result type & contravariant in argument type(parameters)
```scala
trait Function1[-T, +U] {
    def apply(x: T): U
}
trait List[+T] {
    def prepend          (elem: T): List[T] = new Cons(elem, this) //spatne
    def prepend [U >: T] (elem: U): List[U] = new Cons(elem, this) //lower bound
}
```
# IVb Decomposition
 * classification(isSum,isNumber,...)+accessor methods //quadratic increase of methods
 * x.isInstanceOf[T] x.asInstanceOf[T] //unsafe.low level
 * OO Decomposition //nefunguje vzdy, potrebuje menit vsechny vlassy pri nove metode
    ```scala
    trait Expr {
        def eval: Int
        def show:String //musi se implementovat v kazdem podtypu a v nadtypu taky
    }
    class Number(n: Int) extends Expr {
        def eval: Int = n
    }
    class Sum(e1: Expr, e2: Expr) extends Expr {
        def eval: Int = e1.eval + e2.eval
    }
    ```
 * Functional Decomposition with case class and e.match case className
     ```scala
     trait Expr
        case class Number(n: Int) extends Expr
        case class Sum(e1: Expr, e2: Expr) extends Expr
        case class Prod(e1: Expr, e2: Expr) extends Expr
    Object Number {def apply(n: Int) = new Number(n)}
    object Sum {def apply(e1: Expr, e2: Expr) = new Sum(e1, e2)}
    object Prod {def apply(e1: Expr, e2: Expr) = new Prod(e1, e2)}
    def eval(e: Expr): Int = e match {
        case Number(n) => n
        case Sum(e1, e2) => eval(e1) + eval(e2)
    }    
    trait Expr {
        def eval: Int = this match {
            case Number(n) => n  /forward definition????
            case Sum(e1, e2) => e1.eval + e2.eval
            case Prod(e1,e2) => e1.eval + e2.eval
            case Var(x) => ???
        }
        def show(e: Expr): String = e match {
            case Number(n) => x.toString
            case Sum(l,r)=> show(l) + "+" + show(r)
            //TODO addVar Prod and show right parens!!!
            case Sum(e1, e2) => "(" + e1.show + " + " + e2.show + ")"
            case Prod(e1, e2) => e1.show + " * " + e2.show
            case Var(x) => x            
        }
    }    
 ```


# V Lists
val nums = List(1, 2, 3, 4)
val diag3 = List(List(1, 0, 0), List(0, 1, 0), List(0, 0, 1))
:: (pronouced CONS)
nums = 1 :: (2 :: (3 :: (4 :: Nil)))
empty = Nil
A :: B :: C interpreted as A :: (B :: C) 
val nums = 1 :: 2 :: 3 :: 4 :: Nil
val nums = 1 :: (2 :: (3 :: (4 :: Nil)))
:: = prepend
Nil.::(4).::(3).::(2).::(1)
operace na listu head tail isEmpty
PatternMAtching 1 :: 2 :: xs
x :: y :: List(xs, ys) :: zs X ... length >=3

```scala
def isort(xs: List[Int]): List[Int] = xs match {
    case List() => List()
    case y :: ys => insert(y, isort(ys))  //nejcastejsi decompozice listu
}
def insert(x: Int, xs: List[Int]): List[Int] = xs match {
    case List()  => List(x)
    case y :: ys => if (x<=y) x::xs else y::insert(x,ys)
}
```




# Clojure

boot -d seancorfield/boot-new new -n app01
dependency do .boot.profile nebo naopak nefunguje


TEDNE: asi by se to hodilo pro strukturu funsetu, ale zatim staci jiny namespace
clojure: type vs record vs map
deftype mutable defrecord inmutablemaplike
http://clojure.org/reference/datatypes ...why 2?

```clojure
(defprotocol Expression (evaluate [e env] ))
(deftype Number1 [x])
(deftype Add [x y] )
(deftype Multiply [x y])
(deftype Variable [x])
(extend-protocol Expression
  Number1  (evaluate [e env] (.x e ))
  Add      (evaluate [e env] (+ (evaluate (.x e) env) (evaluate (.y e) env)))
  Multiply (evaluate [e env] (* (evaluate (.x e) env) (evaluate (.y e) env)))
  Variable (evaluate [e env] (env (.x e))))
(def environment {"a" 3, "b" 4, "c" 5})
(def expression-tree (Add. (Variable. "a") (Multiply. (Number1. 2) (Variable. "b"))))
(def result (evaluate expression-tree environment))
;---multimethods
(defmulti evaluate (fn [_ [sym _ _]] sym))
    (defmethod evaluate 'Number   [_   [_ x _]] x)
    (defmethod evaluate 'Add      [env [_ x y]] (+ (evaluate env x) (evaluate env y)))
    (defmethod evaluate 'Multiply [env [_ x y]] (* (evaluate env x) (evaluate env y)))
    (defmethod evaluate 'Variable [env [_ x _]] (env x))
(def environment {"a" 3, "b" 4, "c" 5})
(def expression-tree '(Add (Variable "a") (Multiply (Number 2) (Variable "b"))))
(def result (evaluate environment expression-tree))
;--- eval
(defn evaluate [tree env] (eval (clojure.walk/prewalk-replace {'add +, 'multiply *, 'number identity, 'variable env} tree)))
(def environment {:a 3, :b 4, :c 5})
(def expression-tree '(add (variable :a) (multiply (number 2) (variable :b))))
(evaluate expression-tree environment)
;---shorter (fce)
(defn Add [x y] #(+ (x %) (y %)))
(defn Mul [x y] #(* (x %) (y %)))
(defn Var [x] #(x %))
(defn Num [x] (fn [_] x))
(def environment '{a 3 b 4 c 5})
(def expression-tree (Add (Var 'a) (Mul (Num 2) (Var 'b))))
(def result (expression-tree environment))
```
--
defprotocol
extends, extend-type,extend-protocol i existujici typy String/nil
reify
---
boot=
~userhome/.boot/.profile.boot
    (set-env! :local-repo "c:/moje/progs/maven_repo")
nove verze stahovat zde  (https://github.com/boot-clj/boot/releases/download/2.6.0/boot.jar)
    boot.exe nemusi byt v systemroot ve windows staci to pustit odkudkoli a exac stahne novy boot.jar a umistiho do cety= muzu to udelat rucne!
    
