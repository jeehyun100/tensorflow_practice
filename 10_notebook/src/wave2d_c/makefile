.PHONY : clean

CC = gcc
CFLAGS = -O3 -W -Wall

update.so : update.o
	$(CC) -shared -o update.so update.o

update.o : update.c
	$(CC) $(CFLAGS) -c -fPIC update.c -o update.o

clean :
	rm -f update.so update.o
