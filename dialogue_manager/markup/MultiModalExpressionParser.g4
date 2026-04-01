// Grammar for multi-modal expressions -- REFERENCE ONLY; the parser in parser.py is fully stand-alone, and does not rely on this grammar
parser grammar MultiModalExpressionParser;

options {tokenVocab = MultiModalExpressionLexer;}

expression  : ( sentence | action )* ;
sentence    : ( text | variableText )+ ;
action      : ('<' startVerb ident arguments? timeout? '>')    # StartAction
            | ('<' stopVerb ident timeout? '>')                # StopAction
            | ('<' PAUSE '(' number ')' '>')                # PauseAction
            ;

text        : TEXT ;

variableText    : AT query COLUMN string ;
query           : ident (DOT ident)? ;

arguments       : '(' (positionalArgs (',' keywordArgs)? | keywordArgs)? ')' ;
positionalArgs  : value (',' value)* ;
keywordArgs     : keywordArg (',' keywordArg)* ;
keywordArg      : ident '=' value ;

value           : staticValue | variableValue ;
variableValue   : AT query (COLUMN staticValue)? ;
staticValue     : array | literal | string | number ;

array       : '[' value (',' value)* ','? ']' | '[' ']' ;
string      : STRING | ID | TIMEOUT | TTS | verb ;
number      : NUMBER ;

ident       : ID | TIMEOUT | TTS ;
literal     : TRUE | FALSE | NULL ;

verb        : startVerb | stopVerb ;
startVerb   : SET | START | DO ;
stopVerb    : STOP | WAIT ;

timeout     : TIMEOUT '=' NUMBER ;
