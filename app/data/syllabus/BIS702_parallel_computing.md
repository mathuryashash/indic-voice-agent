# BIS702 Parallel Computing (official VTU 2022-scheme syllabus, source https://vtu.ac.in/pdf/2022_3to8/7issyll.pdf)

Semester VII. Course type IPCC (theory + integrated lab). L:T:P:S 3:0:2:0, 40 hours theory + 8-10 lab slots. Credits: 4. CIE 50 marks, SEE 50 marks, exam 3 hours. The syllabus PDF prints the code as BCS702. The theory portion of the IPCC is assessed in both CIE and SEE; the practical (lab) portion is CIE only, so there is no separate external lab exam. SEE questions may include the practical component.

## Module 1: Parallel hardware and software
Introduction to parallel programming, Parallel hardware and parallel software – Classifications of parallel computers, SIMD systems, MIMD systems, Interconnection networks, Cache coherence, Shared-memory vs. distributed-memory, Coordinating the processes/threads, Shared-memory, Distributed-memory.

## Module 2: GPU programming and performance
GPU programming, Programming hybrid systems, MIMD systems, GPUs, Performance – Speedup and efficiency in MIMD systems, Amdahl’s law, Scalability in MIMD systems, Taking timings of MIMD programs, GPU performance.

## Module 3: MPI
Distributed memory programming with MPI – MPI functions, The trapezoidal rule in MPI, Dealing with I/O, Collective communication, MPI-derived datatypes, Performance evaluation of MPI programs, A parallel sorting algorithm.

## Module 4: OpenMP
Shared-memory programming with OpenMP – openmp pragmas and directives, The trapezoidal rule, Scope of variables, The reduction clause, loop carried dependency, scheduling, producers and consumers, Caches, cache coherence and false sharing in openmp, tasking, tasking, thread safety.

## Module 5: CUDA
GPU programming with CUDA - GPUs and GPGPU, GPU architectures, Heterogeneous computing, Threads, blocks, and grids Nvidia compute capabilities and device architectures, Vector addition, Returning results from CUDA kernels, CUDA trapezoidal rule I, CUDA trapezoidal rule II: improving performance, CUDA trapezoidal rule III: blocks with more than one warp.

## Lab experiments (practical component, CIE only)
1 Write a OpenMP program to sort an array on n elements using both sequential and parallel mergesort(using Section). Record the difference in execution time. 2 Write an OpenMP program that divides the Iterations into chunks containing 2 iterations, respectively (OMP_SCHEDULE=static,2). Its input should be the number of iterations, and its output should be which iterations of a parallelized for loop are executed by which thread. For example, if there are two threads and four iterations, the output might be the following: a. Thread 0 : Iterations 0 −− 1 b. Thread 1 : Iterations 2 −− 3 3 Write a OpenMP program to calculate n Fibonacci numbers using tasks. 4 Write a OpenMP program to find the prime numbers from 1 to n employing parallel for directive. Record both serial and parallel execution times. 5 Write a MPI Program to demonstration of MPI_Send and MPI_Recv. 6 Write a MPI program to demonstration of deadlock using point to point communication and avoidance of deadlock by altering the call sequence 7 Write a MPI Program to demonstration of Broadcast operation. 8 Write a MPI Program demonstration of MPI_Scatter and MPI_Gather 9 Write a MPI Program to demonstration of MPI_Reduce and MPI_Allreduce (MPI_MAX, MPI_MIN, MPI_SUM, MPI_PROD)

## Textbooks and references
1. Peter S Pacheco, Matthew Malensek – An Introduction to Parallel Programming, second edition, Morgan Kauffman. 2. Michael J Quinn – Parallel Programming in C with MPI and OpenMp, McGrawHill. Reference Books: 1. Calvin Lin, Lawrence Snyder – Principles of Parallel Programming, Pearson 2. Barbara Chapman – Using OpenMP: Portable Shared Memory Parallel Programming, Scientific and Engineering Computation 3. William Gropp, Ewing Lusk – Using MPI:Portable Parallel Programing, Third edition, Scientific and Engineering Computation
