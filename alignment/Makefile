CPP = g++ -Wall -std=c++11
CPPFLAGS = -O2

all:
	@echo "Please give an option."
fit:
	@g++ -Wall -std=c++11 -I/opt/homebrew/Cellar/gsl/2.7/include -c RotTransFit.cpp
	@g++ -L/opt/homebrew/Cellar/gsl/2.7/lib -o RotTransFit RotTransFit.o -lgsl -lgslcblas
clean:
	@rm RotTransFit RotTransFit.o
