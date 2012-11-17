// RUN: -UFOO

#ifdef FOO
// this will be removed
#endif

#ifdef BAR
// this will not be removed
#endif
