CC = icpc
CFLAGS = -O3 -Wall

generator.run: generator/main.cpp Makefile
	$(CC) $(CFLAGS) generator/main.cpp -o generator.run

viewer.run: viewer/main.cpp Makefile
	$(CC) $(CFLAGS) viewer/main.cpp -o viewer.run

matrix: generator.run
	./generator.run 500 a.mat 1994
	./generator.run 500 b.mat 1995

clean:
	rm ./*.run
