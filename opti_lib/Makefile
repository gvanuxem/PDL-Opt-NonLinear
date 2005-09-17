CC = gcc
F77 = g77
CFLAGS = -O3 -fPIC

all:    asa.o  asa_usr.o  de36.o  dhc.o  hooke.o \
        cgfam.o  cgsearch.o  daxpy.o  dnrm2.o  dxpy.o \
        gencan.o  lbfgsb.o  lbfgs.o  spg.o  tensor.o  tn.o \
        asa.o  asa_usr.o  de36.o dhc.o  hooke.o milalonga.o
	$(CC) -shared -fPIC -lm -lg2c -o libopti.so \
        asa.o  asa_usr.o de36.o  dhc.o  hooke.o \
        cgfam.o  cgsearch.o  daxpy.o  dnrm2.o  dxpy.o \
        gencan.o  lbfgsb.o  lbfgs.o  spg.o  tensor.o  tn.o \
        milalonga.o


cgfam.o : cgfam.f
	$(F77) $(CFLAGS) -c cgfam.f

cgsearch.o : cgsearch.f
	$(F77) $(CFLAGS) -c cgsearch.f

daxpy.o : daxpy.f
	$(F77) $(CFLAGS) -c daxpy.f

dnrm2.o : dnrm2.f
	$(F77) $(CFLAGS) -c dnrm2.f

dxpy.o : dxpy.f
	$(F77) $(CFLAGS) -c dxpy.f

gencan.o : gencan.f
	$(F77) $(CFLAGS) -c gencan.f

lbfgsb.o : lbfgsb.f
	$(F77) $(CFLAGS) -c lbfgsb.f

lbfgs.o : lbfgs.f
	$(F77) $(CFLAGS) -c lbfgs.f

spg.o : spg.f
	$(F77) $(CFLAGS) -c spg.f

tensor.o : tensor.f
	$(F77) $(CFLAGS) -c tensor.f

tn.o : tn.f
	$(F77) $(CFLAGS) -c tn.f

milalonga.o : milalonga.f
	$(F77) $(CFLAGS) -c milalonga.f

asa.o : asa.c asa.h  asa_usr_asa.h
	$(CC) $(CFLAGS) -c asa.c
	
asa_usr.o : asa_usr.c asa.h  asa_usr_asa.h
	$(CC) $(CFLAGS) -c asa_usr.c

de36.o : de36.c de.h
	$(CC) $(CFLAGS) -c de36.c

dhc.o : dhc.c
	$(CC) $(CFLAGS) -c dhc.c

hooke.o : hooke.c
	$(CC) $(CFLAGS) -c hooke.c

clean : 
	rm *.o libopti.so

