use warnings;
use strict;

use Debian::Debhelper::Dh_Lib;

insert_after('dh_installdocs', 'dh_doxygen');

1;
