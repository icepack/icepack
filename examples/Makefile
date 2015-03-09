
include $(DEAL_II_DIR)/common/Make.global_options

INCLUDE += -I../include

poisson_basic: poisson_basic.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(LIBS) $(CXXFLAGS.g)
