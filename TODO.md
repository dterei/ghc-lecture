# Improvements

## Polymorphic Functions

Worth spending some time on two different approaches to polymorphic functions?

1. First-class: uniform data representation (pointers), one code path.
2. Second-class (macros): code replicated and specialized to each type, value
   representation can differ.

Haskell uses (1) while C++ (templates, overloading) uses (2). Optimizations
push (1) more towards (2), (2) can be seen as a partial-evaluation of (1).

This is perhaps a good motivator for the overall design: everything is a
pointer/closure! And allows us to bring in unboxed types later. Should also
mention trade-offs, (2) eleminates all costs but can't handle seperate
compilation and cannot be done for polymorphically recursive or higher-rank
functions.

## Typeclasses

Incorporate a recursive example where we need to build a new dictionary
dynamically:

    print_n :: Show a => Int -> a -> IO ()
    print_n 0 a = print a
    print_n n a = print_n (n-1) (replicate n a)

Taken from: http://okmij.org/ftp/Computation/typeclass.html

May also be worth pointing out the benefits of type-classes over manual
translation, that is, type-directed inference to save a lot of coding. Hiding
the plumbing is what makes type-classes so useful and popular.

