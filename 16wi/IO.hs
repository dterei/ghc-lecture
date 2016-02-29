module IO where

import System.IO

data T = T Int Int

compIO :: Handle -> IO ()
compIO h =
  do name <- hGetLine h
     hPutStrLn h name

mmap :: (a -> b) -> [a] -> [b]
mmap f []     = []
mmap f (x:xs) = f x : map f xs

mkT :: Int -> Int -> T
mkT = T

myid = id
