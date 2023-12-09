- [Assignment PDF](./A1.pdf)
- [Compile Script](compile.sh)
- [Contributions File](contributions.txt)
- [Interface Script](interface.sh)
- [Implementation of FPTree](fptree.cpp)
- [Include Header for FPTree](fptree.hpp)
- [Transactional DB Compression](main.cpp)
- [Decompression Algorithm](decompression.cpp)
- [Verfication Script](verify.cpp)

Compile:
```
bash compile.sh
```

Compress:
```
bash interface.sh C <path/to/dataset.dat> <path/to/output.dat>
```

Decompress:
```
bash interface.sh D <path/to/compressed_dataset.dat> <path/to/reconstructed.dat>
```