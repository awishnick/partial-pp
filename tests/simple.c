// RUN: -DTESTSYM
#ifdef TESTSYM
// this is defined
#endif

#ifdef NOT_TESTSYM
// this is not defined
#endif
