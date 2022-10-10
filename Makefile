CXX=g++
CXXFLAGS=-Ixbyak
LDFLAGS=

all: test2.cpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS)

dis: dump.obj
	./dis.sh $<

clean:;
	rm -rf a.out *.o *~ dump.obj


