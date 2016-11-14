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
Dalsi metody:
 * length
 * last (opak head)   slozitost n |xs|
 * init (opak rest/tail)
 * take n
 * drop n
 * xs (n)
 * xs ++ ys    concat complexity=|xs|
 * reverse          complexity n*n (later better impl)
 * updated (n,x)
 * indexOf x
 * contains x
 * xs ::: ys     ys.:::(xs) =prepend listu na listu
 * xs splitAt n  // vraci tuple

Rozdil mezi ::: concat a ++???? 

```scala
def last[T](xs: List[T]): T = xs match {
    case List() => throw new Error("last of empty list")
    case List(x) => x
    case y :: ys => last(ys)
}
def init[T](xs: List[T]): List[T] = xs match {
    case List() => throw new Error("init of empty list")
    case List(x) => List() //doplneno
    case y :: ys => y :: init (ys) //dopl, case se provadi v poradi v jakem jsou napsane
}
def concat[T](xs: List[T], ys: List[T]) = xs match {
    case List() => ys
    case z :: zs => z :: concat(zs, ys)  //complexity |xs|    
}
def reverse[T](xs: List[T]): List[T] = xs match {
    case List() => List()
    case y :: ys => reverse(ys) ++ List(y)
}
def removeAt[T](xs: List[T], n: Int) = (xs take n) ::: (xs drop n+1)

//TODO
def flatten(xs: List[Any]): List[Any] = ???
flatten(List(List(1, 1), 2, List(3, List(5, 8))))   > res0: List[Any] = List(1, 1, 2, 3, 5, 8)
```
merge sort
    1=setrideny
    >1=rozdelnapul,setrid pulky, spoj(se zatridovanim???nebone??) odopoved fvefci merge
```scala
def msort(xs: List[Int]): List[Int] = {
    val n = xs.length/2
    if (n == 0) xs
    else {
        def merge(xs: List[Int], ys: List[Int]) = ???
        val (fst, snd) = xs splitAt n
        merge(msort(fst), msort(snd))
    }
}
def merge(xs: List[Int], ys: List[Int]) = xs match { //improved Later    
    case Nil =>  ys
    case x :: xs1 => ys match {
        case Nil => xs
        case y :: ys1 => if (x < y) x :: merge(xs1, ys) else y :: merge(xs, ys1)
    }
}
//LEPSI!!!! s pair pattern matching
def merge(xs: List[Int], ys: List[Int]): List[Int] = (xs, ys) match {
    case (Nil,ys) => ys
    case (xs,Nil) => xs
    case (x::xs1, y:ys1) => if (x<y) x::merge(xs1,ys) else yy :: merge(xs,ys1)
}
```
* Pair (x1,x2)
* Tuple (x1,x2,...xN)
        case class Tuple2[T1, T2](_1: +T1, _2: +T2) {
            override def toString = ”(” + _1 + ”,” + _2 +”)”
        }
        val (label, value) = pair
        val label = pair._1
        val value = pair._2

## Implicit Parameters

``scala        
def msort[T](xs: List[T])(lt: (T, T) => Boolean) = {
    ...
    merge(msort(fst)(lt), msort(snd)(lt))
} 
if (lt(x, y)) {}//v merge emtode misto <
//Calls
val xs = List(-5, 6, 3, 2, 7)
val fruit = List(”apple”, ”pear”, ”orange”, ”pineapple”)
merge(xs)((x: Int, y: Int) => x < y)    
merge(fruit)((x: String, y: String) => x.compareTo(y) < 0)


//pouziti misto toho: scala.math.Ordering[T]
def msort[T](xs: List[T])(ord: Ordering) =
    def merge(xs: List[T], ys: List[T]) =
    ... if (ord.lt(x, y)) ...
merge(msort(fst)(ord), msort(snd)(ord))     
//!!! ord.lt(x,y)
//calls
msort(nums)(Ordering.Int)
msort(fruits)(Ordering.String)

def msort[T](xs: List[T])(implicit ord: Ordering) =
msort(nums)  //najde to z typu parrametru compiler
msort(fruits)
```

## higher order functions
xs map (x => x * factor)
xs filter (x => x > 0)
xs filterNot p
xs partition p (filter/not in single traversal)
xs takeWhile p
xs dropWhile p
xs span p  (split-with pred coll)

### reduce/folder
def sum(xs: List[Int]) = (0 :: xs) reduceLeft ((x, y) => x + y)
def product(xs: List[Int]) = (1 :: xs) reduceLeft ((x, y) => x * y)
def sum(xs: List[Int]) = (0 :: xs) reduceLeft (_ + _) //every _ represents new parameter from left to right
def product(xs: List[Int]) = (1 :: xs) reduceLeft (_ * _)
foldleft s accumuatorem
def sum(xs: List[Int]) = (xs foldLeft 0) (_ + _)
def product(xs: List[Int]) = (xs foldLeft 1) (_ * _)
```scala
abstract class List[T] { //mozna implementace
    def reduceLeft(op: (T, T) => T): T = this match {
        case Nil => throw new Error("Nil.reduceLeft")
        case x :: xs => (xs foldLeft x)(op)
    }
    def foldLeft[U](z: U)(op: (U, T) => U): U = this match {
        case Nil => z
        case x :: xs => (xs foldLeft op(z, x))(op)
    }
}
```

list.reduceLeft op = (+ (+ (+ x1 x2) x3) ...xN)
list.reduceRight op = (+ x1 (+ x2 ..))
(List(x1, ..., xn) foldRight acc)(op)
```scala
def reduceRight(op: (T, T) => T): T = this match {
    case Nil => throw new Error("Nil.reduceRight")
    case x :: Nil => x
    case x :: xs => op(x, xs.reduceRight(op))
}
def foldRight[U](z: U)(op: (T, U) => U): U = this match {
    case Nil => z
    case x :: xs => op(x, (xs foldRight z)(op))
}
```
pro associative a commutative je to stejne nekdy to nejde (treba :: s emusi volat na listu a ne na prvku
```scala
    def concat[T](xs: List[T], ys: List[T]): List[T] = (xs foldRight ys) (_ :: _)
///Ex: def reverse[a](xs: List[T]): List[T] = (xs foldLeft List[T]())((xs, x) => x :: xs)    //typ List[T]() je potreba pro type inference(to je zas co?)
///TODO:
def mapFun[T, U](xs: List[T], f: T => U): List[U] = (xs foldRight List[U]())( ??? )
def lengthFun[T](xs: List[T]): Int = (xs foldRight 0)( ??? )
```

# VI Collections
List: linear access
Vector: very shallow tree: more balanced access than list 32, 232*32=2^10,2^15,2^20,...
val nums = Vector(1, 2, 3, -88)
val people = Vector(”Bob”, ”James”, ”Peter”)
Metody jako na Listu krome:
    * x +: xs   //x na zacatku
    * xs :+ x   //x na konci  :je tam,kde je collection
    * complecity pridavani log32(N) object creation
CollectionClasses:
    * List Vector  (String Array Range) allsubcla of Seq  ,IndexedSequence(Vectgor,Range)
    * Seq(uence),Set,Map
    * Iterable
Range
```scala
val r: Range = 1 until 5  //1,2,3,4
val s: Range = 1 to  5    //1,2,3,4,5
            1 to 10 by 3  //1,4,7,10
            6 to 1  by -2 //6,4,2
```
Sequence Operations
* xs exists p       true if there is an element x of xs such that p(x) holds,false otherwise.
* xs forall p       true if p(x) holds for all elements x of xs , false otherwise.
* xs zip ys         A sequence of pairs drawn from corresponding elementsof sequences xs and ys .
* xs.unzip          Splits a sequence of pairs xs into two sequences consisting of the first, respectively second halves of all pairs.
* xs.flatMap f      Applies collection-valued function f to all elements of xs and concatenates the results //JAKY JE ROZDIL oproti MAP???
* xs.sum            The sum of all elements of this numeric collection.
* xs.product        The product of all elements of this numeric collection
* xs.max            The maximum of all elements of this collection (an Ordering must exist)
* xs.min            The minimum of all elements of this collection

Examples...
```scala
    (1 to M) flatMap (x => (1 to N) map (y => (x, y)))

    def scalarProduct(xs: Vector[Double], ys: Vector[Double]): Double = (xs zip ys).map(xy => xy._1 * xy._2).sum
    def scalarProduct(xs: Vector[Double], ys: Vector[Double]): Double = (xs zip ys).map{ case (x, y) => x * y }.sum
    ///{ case p1 => e1 ... case pn => en } =====      x => x match { case p1 => e1 ... case pn => en }

    def isPrime(n: Int): Boolean = (2 until n) forall (d=> n%d != 0)
```    

combinatorics/Search
(1 until n) map (i => (1 until i) map (j => (i, j)))
flatten ===== (xss foldRight Seq[Int]())(_ ++ _)
(1 until n) flatMap (i => (1 until i) map (j => (i, j)))  ========== 
((1 until n) map    (i => (1 until i) map (j => (i, j)))).flatten

result=(1 until n) flatMap (i => (1 until i) map (j => (i, j))) filter ( pair => isPrime(pair._1 + pair._2))

### For Expression
for ( s ) yield e // s =sequence of generators and filters
    generator p<-e  //last generator vary fastest
    filter if p
for {s} yield e //generatory nepotrebuji stredniky a muzou byt na vice radcich    
```scala
        case class Person(name: String, age: Int)
        for ( p <- persons if p.age > 20 ) yield p.name
        persons filter (p => p.age > 20) map (p => p.name)

flatmap priklad
    for {
        i <- 1 until n
        j <- 1 until i
        if isPrime(i + j)
    } yield (i, j)        

//ExampleDalsi    
def scalarProduct(xs: List[Double], ys: List[Double]) : Double =    (for ((x,y)<-xs zip ys) yield x*y).sum
```

### Sets (6.3)
```scala
val fruit = Set(”apple”, ”banana”, ”pear”)
val s = (1 to 6).toSet
s map (_ + 2)
fruit filter (_.startsWith == ”app”)
s.nonEmpty
///Iterables: co vsechno jde na setu volat
```
 * unordered
 * no duplicate elements ... s map (_ / 2) // Set(2, 0, 3, 1)
 * s contains 5 // true
Example 
```scala
def queens(n: Int) = {
    def placeQueens(k: Int): Set[List[Int]] = {
        if (k == 0) Set(List())
        else for {
                queens <- placeQueens(k - 1)
                    col <- 0 until n
                    if isSafe(col, queens)
            } yield col :: queens
    }
    placeQueens(n)
}
def isSafe(col: Int, queens: List[Int]): Boolean = {
  val row =queens.length
  //List(0,3,1)=>List((2,0),(1,3),(0,1))
  val queensWithRows = (row-1 to 0 step -1).zip queens
  queensWithRows.forAll { case (r,c) => col!=c  &&  match.abs(col-c)!=row-r //diagonala
  }
}
def show(queens:List[Int])={
  val lines= for (col< queens.reverse) yield Vector.fill(queens.length)("* ").updated(col,"X ").mkString
  "\n" + (lines mkString "\n")
}
queens(4) map show
(queens(8) take 3 map show) mkString "\n"
```
### Maps(6.4)

```scala
al romanNumerals = Map("I" -> 1, "V" -> 5, "X" -> 10)
val capitalOfCountry = Map("US" -> "Washington", "Switzerland" -> "Bern")
val countryOfCapital = capitalOfCountry map {case(x, y) => (y, x)}

capitalOfCountry("andorra")
capitalOfCountry get "andorra"  
//Some(Washington),None :Option

trait Option[+A]
case class Some[+A](value: A) extends Option[A]
object None extends Option[Nothing]

def showCapital(country: String) = capitalOfCountry.get(country) match {
  case Some(capital) => capital
  case None => "missing data"
}

val fruit = List("apple", "pear", "orange", "pineapple")
fruit sortWith (_.length < _.length) // List("pear", "apple", "orange", "pineapple")
fruit.sorted // List("apple", "orange", "pear", "pineapple")

//groupBy=discriminator function
fruit groupBy (_.head) //> Map(p -> List(pear, pineapple),  //| a -> List(apple),  //| o -> List(orange))


class Poly(terms0: Map[Int, Double]) {
  def this(bindings: (Int, Double)*) = this(bindings.toMap)  //repeated parameter!!!!!!!
  val terms = terms0 withDefaultValue 0.0   //total function, nehazi vyjimku nikdy, misto toho poskytuje 0
  //++ def + (other: Poly) = new Poly(terms ++ (other.terms map adjust))  //to scala nema merge-with????
  def + (other: Poly) = new Poly((other.terms foldLeft terms) (addTerm))
  def addTerm(terms: Map[terms:Map[Int,Double],term:(Int,Double)): Map[Int,Double] ={
    val (exp,coeff)=term
    terms + (exp -> (coeff + terms(exp)))
  }
  def adjust(term: (Int, Double)): (Int, Double) = {
    val (exp, coeff) = term
    /*
    terms get exp match {
      case Some(coeff1)=>   exp -> (coeff + coeff1)
      case None =>          exp -> coeff   // x->y je syntactic sugar pro (x,y) Pair(x,y)
    }
    */
    exp -> (coeff + terms(exp))  // s defualt hodnotama se to zjednodusi
  }
  override def toString = (for ((exp, coeff) <- terms.toList.sorted.reverse) yield coeff+"x^"+exp) mkString " + "
}
```

### posledni priklad 6.5 Phonebook 2000
```scala
val mnemonics = Map(’2’ -> ”ABC”, ’3’ -> ”DEF”, ’4’ -> ”GHI”, ’5’ -> ”JKL”,’6’ -> ”MNO”, ’7’ -> ”PQRS”, ’8’ -> ”TUV”, ’9’ -> ”WXYZ”)

val mnem = Map('2' -> "ABC", '3' -> "DEF", '4' -> "GHI", '5' -> "JKL",'6' -> "MNO", '7' -> "PRQS", '8' -> "TUV", '9' -> "WXYZ")

val in = Source.fromURL("http://lamp.epfl.ch/files/content/sites/lamp/files/teaching/progfun/linuxwords.txt")
val words = in.getLines.toList filter(w => w forall(chr => chr.isLetter))

val charCode: Map[Char, Char] = for ((digit, str) <- mnem; ltr <- str) yield ltr -> digit
def wordCode(word: String): String = word.toUpperCase map charCode
//wordCode("JAVA")
//wordCode("Java")
val wordsForNum: Map[String, Seq[String]] = words groupBy wordCode withDefaultValue Seq()
//wordsForNum("JAVA")
def encode(number: String): Set[List[String]] ={
    if (number.isEmpty) Set(List())
    else
      { for {
        split <- 1 to number.length        // find out what first word must be
        word <- wordsForNum(number take split)
        rest <- encode(number drop split)
      } yield word :: rest
    } toSet
  }
//encode("7225247386")
def translate(number: String): Set[String] = encode(number). map(_ mkString " ")
//translate("7225247386")
type Word = String
type Sentence = List[Word]
type Occurrences = List[(Char, Int)]
def wordOccurrences(w: Word): Occurrences =
    w.groupBy(c => c.toLower)
     .map(m => (m._1, m._2.length)).toList      //    w.groupBy(char => w.count(char))      
    dictionary.map(word => wordOccurrences(word), word)  
}
```


* ScalaCheatsheet LaurentPoulain  ?https://gist.github.com/jaturken/3976117
* Scala school by twiter
* Scala Exercises
* Book(programming inScala)
* scala web site/scaladoc

Not
* larger context
* FP desing principles
* State, mutable, what changes if we add it, pure functions??/
* paral&distrib system: user immutability, dist.coll and big data





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
    
