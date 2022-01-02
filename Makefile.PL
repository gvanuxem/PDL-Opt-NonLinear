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

my @pack = ([qw(nonlinear.pd NonLinear PDL::Opt::NonLinear)]);
my @opti_libfiles = map {s/^opti_lib\///; s/\.f$//; $_} glob("opti_lib/*.f");

my %hash = pdlpp_stdargs(@pack);
$hash{INC} .= " -Iopti_lib";  # uncomment as required
$hash{OBJECT} .= " opti_lib/hooke\$(OBJ_EXT) opti_lib/asa\$(OBJ_EXT) opti_lib/asa_usr\$(OBJ_EXT) opti_lib/dhc\$(OBJ_EXT) opti_lib/de36\$(OBJ_EXT) ";
my $fobj = join '', map {" opti_lib/$_\$(OBJ_EXT) "} @opti_libfiles;
$hash{OBJECT} .= $fobj; 
$hash{LDFROM} .= " hooke\$(OBJ_EXT) asa\$(OBJ_EXT) asa_usr\$(OBJ_EXT) dhc\$(OBJ_EXT) de36\$(OBJ_EXT) ".$fobj." NonLinear\$(OBJ_EXT)";
$hash{LIBS}[0] .= $f77->runtime ;
$hash{LIBS}[0] .= ' -L/usr/local/lib -lport -lblas' ;
$hash{clean}{FILES} .= " f77_underscore" .
  join '', map {" opti_lib/$_\$(OBJ_EXT) "} @opti_libfiles;
$hash{clean}{FILES} .= " hooke\$(OBJ_EXT) ";

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
	'BUILD_REQUIRES' =>  { "PDL" => 0, "PDL::LinearAlgebra" => 0,},
	'VERSION' => "0.05",
	'PREREQ_PM' => {
	  'PDL' => 0, 
	  'PDL::LinearAlgebra' => 0,
	},
);

sub MY::postamble {
        my $mycompiler     = $f77->compiler();
        my $mycflags       = $f77->cflags();
	my $orig = pdlpp_postamble(@pack);
	$orig =~ s/:\s*nonlinear\.pd/: nonlinear.pd/;
	$orig .join "\n",map {
("
opti_lib/$_\$(OBJ_EXT): opti_lib/$_.f 
	$mycompiler -c -o opti_lib/$_\$(OBJ_EXT) $mycflags -O3 -fPIC opti_lib/$_.f
" )} @opti_libfiles;
}
