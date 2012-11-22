// RUN: -DDEFINED -UUNDEFINED
#if defined(DEFINED)
// this is defined
#endif

#if defined(UNDEFINED)
// this is undefined
#endif

#if defined(FOO)
// this is unknown and will be left
#endif
