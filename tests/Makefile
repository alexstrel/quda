include ../make.inc

QUDA = ../lib/libquda.a

INC += -I../include -I.

HDRS = blas_reference.h dslash_reference.h test_util.h

TESTS = dslash_test invert_test su3_test pack_test blas_test

all: $(TESTS)

dslash_test: dslash_test.o test_util.o dslash_reference.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@

invert_test: invert_test.o test_util.o dslash_reference.o blas_reference.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@

su3_test: su3_test.o test_util.o dslash_reference.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@

pack_test: pack_test.o test_util.o dslash_reference.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@

blas_test: blas_test.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@

tune: blas_test
	./blas_test $(DEVICE) && cp -f blas_param.h ../lib/

clean:
	-rm -f *.o $(TESTS)

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) $< -c -o $@

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) $< -c -o $@

%.o: %.cu $(HDRS)
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

.PHONY: all tune clean
