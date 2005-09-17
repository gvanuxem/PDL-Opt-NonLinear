
# Use this as a template for the Makefile.PL for
# any external PDL module.

use ExtUtils::MakeMaker;

use PDL::Core::Dev;
@pack = (["nonlinear.pd",NonLinear,PDL::Opt::NonLinear]);

%hash = pdlpp_stdargs(@::pack);
# $hash{'OPTIMIZE'}  = '-g'; # If you want to debug, uncomment this.
$hash{INC} .= " -Iopti_lib";  # uncomment as required
#$hash{LIBS}[0] .= " -L/usr/local/lib -lport -lopti -lg2c"; # uncomment as required

#$hash{LDLOADLIBS} .= 'oldnames.lib kernel32.lib user32.lib gdi32.lib winspool.lib  comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib  netapi32.lib uuid.lib wsock32.lib mpr.lib winmm.lib  version.lib odbc32.lib odbccp32.lib msvcrt.lib  opti_lib/libport3.lib opti_lib/liboptim.lib' if $^O =~ /MSWin/;

$hash{LIBS}[0] .= " -L/usr/local/lib -lopti -lg2c"; # uncomment as required

$hash{LDLOADLIBS} .= 'oldnames.lib kernel32.lib user32.lib gdi32.lib winspool.lib  comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib  netapi32.lib uuid.lib wsock32.lib mpr.lib winmm.lib  version.lib odbc32.lib odbccp32.lib msvcrt.lib  opti_lib/liboptim.lib' if $^O =~ /MSWin/;


WriteMakefile(%hash,
	VERSION => 0.01,
);

# Add genpp rule
# add other makefile additions as required (see also ExtUtils::MakeMaker)
sub MY::postamble {
	pdlpp_postamble(@::pack);
}

