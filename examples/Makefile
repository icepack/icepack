
include $(DEAL_II_DIR)/common/Make.global_options

INCLUDE += -I../include

TARGETS = ssa
LIBS += -L../build/lib/ -licepack

all: $(TARGETS) half_circle.msh

%.msh: %.geo
	gmsh -2 $<

%.geo:
	python make_half_circle.py

%: %.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(LIBS) $(INCLUDE)

clean:
	rm -rf $(TARGETS) *.o *.eps *.vtk
