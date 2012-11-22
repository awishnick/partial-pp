// RUN: -DDEFINED -DYES=1 -UUNDEFINED -DNO=0
#if YES && defined(DEFINED)
// yes
#endif

#if defined(DEFINED) && YES
// yes
#endif

#if defined(UNDEFINED) && YES
// no
#endif

#if YES && NO
// no
#endif

#if defined(NO) && YES
// yes
#endif
