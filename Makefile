CXX=g++-12
CXX=dpcpp
CXX=g++
CXX=clang++-14

CXXFLAGS=-Ixbyak -Iperf -O0
CXXFLAGS=-Ixbyak -Iperf -O3
LDFLAGS=

all: test4.cpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS)

mm: test5.cpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS)

dis: dump.obj
	./dis.sh $<

asm: test4.cpp
	$(CXX) -S $(CXXFLAGS) $< 

clean:;
	rm -rf a.out *.o *.s *~ dump.obj


