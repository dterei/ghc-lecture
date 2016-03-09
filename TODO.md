# Improvements

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

