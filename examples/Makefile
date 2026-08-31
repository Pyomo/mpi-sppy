CC = mpicc
CFLAGS = -O2 -g -Wall -Wextra

TARGET = mpi_rma_get_reproducer

all: $(TARGET)

$(TARGET): $(TARGET).c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
