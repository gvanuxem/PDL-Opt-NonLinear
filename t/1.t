#!/usr/bin/perl
use PDL::LiteF;
use PDL::Opt::NonLinear;
use Test;

BEGIN { plan tests => 1 };

sub fapprox {
	my($a,$b) = @_;
	PDL::abs($a-$b)->max < 0.0001;
}

$res = pdl([1,1,1,1,1]);



$x = random(5);

$gx = rosen_grad($x);
$hx = rosen_hess($x);
$fx = rosen($x);
$xtol = pdl(1e-16);
$gtol = pdl(1e-8);
$stepmx =pdl(0.5);
$maxit = pdl(long, 50);
sub min_func{
	my ($fx, $x) = @_;
	$fx .= rosen($x);
}
sub grad_func{
	my ($gx, $x) = @_;
	$gx .= rosen_grad($x);
}
sub hess_func{
	my ($hx, $x) = @_;
	$hx .= rosen_hess($x);
}
tensoropt($fx, $gx, $hx, $x, 
	  1,$maxit,15,1,2,1,
	  ones(5),0.5,$xtol,$gtol,2,6,
	  \&min_func, \&grad_func, \&hess_func);

ok(fapprox($x,$res));
