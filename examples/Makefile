
include $(DEAL_II_DIR)/common/Make.global_options

INCLUDE += -I../include

TARGETS = poisson_basic

all: $(TARGETS)

%: %.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(LIBS) $(CXXFLAGS.g)

clean:
	rm -rf $(TARGETS) *.o *.eps
