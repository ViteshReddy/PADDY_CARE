%option noyywrap
%{
#include "symptom.tab.h"
%}

%%
[Bb][Rr][Oo][Ww][Nn][ \t]+[Ss][Pp][Oo][Tt][Ss]?  { return BROWN_SPOTS; }
[Bb][Rr][Oo][Ww][Nn]                              { return BROWN; }
[Ss][Pp][Oo][Tt][Ss]?                             { return SPOTS; }
[Yy][Ee][Ll][Ll][Oo][Ww][Ii][Nn][Gg]?             { return YELLOWING; }
[Dd][Rr][Yy][Ii][Nn][Gg]?                         { return DRYING; }
[Bb][Ll][Ii][Gg][Hh][Tt]                          { return BLIGHT; }
[Rr][Uu][Ss][Tt]                                  { return RUST; }
[Ww][Ii][Ll][Tt][Ii][Nn][Gg]?                     { return WILTING; }
[Ll][Ee][Ss][Ii][Oo][Nn][Ss]?                      { return LESIONS; }
[Mm][Ii][Ll][Dd][Ee][Ww]                          { return MILDEW; }
[A-Za-z]+                                          { return OTHER; }
[0-9]+                                            { return OTHER; }
[ \t\r\n.,;:!?()\-]+                              { /* ignore separators */ }
.                                                 { return OTHER; }
%%
