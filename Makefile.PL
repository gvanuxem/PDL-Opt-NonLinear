use strict;
use warnings;
use ExtUtils::MakeMaker;
use PDL::Core::Dev;

my $f77;
BEGIN {
   eval "use ExtUtils::F77";
   if ($@ ne "") {
     warn "\n\tExtUtils::F77 module not found. Will not build PDL::Opt::NonLinear\n\n" ;
     exit(1);
   }
   else {
     $f77 = 'ExtUtils::F77';
   }
   if (!$f77->testcompiler) {
     warn "\n\n\tNo fortran compiler found. Will not build PDL::Opt::NonLinear on this system\n\n";
     exit(1);
   }
}

my @pack = qw(nonlinear.pd NonLinear PDL::Opt::NonLinear);
my @opti_libfiles = map {s/^opti_lib\///; s/\.f$//; $_} glob("opti_lib/*.f");
my $fobj = join ' ', map {"opti_lib/$_\$(OBJ_EXT)"} @opti_libfiles;
my @opti_cfiles = qw(hooke asa asa_usr dhc de36);
my $cobj = join ' ', map "$_\$(OBJ_EXT)", @opti_cfiles;

my %hash = pdlpp_stdargs(\@pack);
$hash{INC} .= " -I.";  # uncomment as required
$hash{OBJECT} .= " $cobj $fobj";
$hash{LIBS}[0] .= $f77->runtime . ' -L/usr/local/lib -lport -lblas' ;
$hash{clean}{FILES} .= " f77_underscore $fobj";

# Create flag file according to whether or not to use
# underscores (pretty hacky)
unlink("f77_underscore") if -e "f77_underscore";
if ($f77->trail_) {
   open OUT, ">f77_underscore" or die "unable to write scratch file";
   close OUT;
}

WriteMakefile(
        %hash,
        'ABSTRACT' => 'Non linear optimization routines for PDL',
        'AUTHOR' => [ 'Gregory Vanuxem <gvanuxem.perl@gmail.com>' ],
        'LICENSE' => 'artistic_2',
	'CONFIGURE_REQUIRES' =>  { "PDL" => 0, "ExtUtils::F77" => 0,},
	'PREREQ_PM' => {
	  'PDL' => 0, 
	  'PDL::LinearAlgebra' => 0,
	},
);

sub MY::postamble {
        my $mycompiler     = $f77->compiler();
        my $mycflags       = $f77->cflags();
	my $orig = pdlpp_postamble(\@pack);
	$orig =~ s/:\s*nonlinear\.pd/: nonlinear.pd/;
	$orig .join "\n",map {
("
opti_lib/$_\$(OBJ_EXT): opti_lib/$_.f
	$mycompiler -c -o opti_lib/$_\$(OBJ_EXT) $mycflags -O3 -fPIC opti_lib/$_.f
" )} @opti_libfiles;
}
