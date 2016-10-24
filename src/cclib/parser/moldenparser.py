# -*- coding: utf-8 -*-

"""Parser for MOLDEN format files

Several programs support writing their resulting wave functions in MOLDEN format [1]_

Especially in connection with TheoDORE [2]_ writing NTOs in this format, one might find cclib
supporting it useful.

.. [1] http://www.cmbi.ru.nl/molden/molden_format.html

.. [2] http://theodore-qc.sourceforge.net/

"""
from __future__ import division
from __future__ import print_function

from pprint import pprint

from . import logfileparser
from . import utils


class MOLDEN(logfileparser.Logfile):
    """A MOLDEN formatted file parser"""

    def __init__(self, *args, **kwargs):
        # Call the __init__ method of the superclass
        super(MOLDEN, self).__init__(logname="MOLDEN", *args, **kwargs)
        self.first_line = None
        self.atoms_section = None
        self.gto_section = None
        self.mo_section = None
        self.coordinate_units_bohr = None

    def __str__(self):
        """Return a string representation of the object."""
        return "MOLDEN formatted file %s" % (self.filename)

    def __repr__(self):
        """Return a representation of the object."""
        return 'MOLDEN("%s")' % (self.filename)

    def before_parsing(self):
        self.first_line = True

    def after_parsing(self):
        pass

    def _extract_atoms(self, inputfile, line):
        if line.startswith('[Atoms]'):
            if line.find('Angs') > 0:
                coordinate_units_bohr = False
            elif line.find('AU') > 0:
                coordinate_units_bohr = True
            else:
                raise RuntimeError('Atoms section expecting coordinate unit Angs|AU')
            if not hasattr(self, "atomcoords"):
                self.atomcoords = []
            atomcoords = []
            atomnos = []
            while True:
                line = next(inputfile)
                if line.startswith('['):
                    break
                element, number, atomic_number, x, y, z = line.split()
                atomnos.append(int(atomic_number))
                if coordinate_units_bohr:
                    atomcoords.append([utils.convertor(float(i), 'bohr', 'Angstrom') for i in (x, y, z)])
                else:
                    atomcoords.append([float(x), float(y), float(z)])
            self.set_attribute('atomnos', atomnos)
            self.set_attribute('natom', len(atomnos))
            self.atomcoords.append(atomcoords)
        return line

    @staticmethod
    def _extract_one_gto(sequence_number, inputfile):
        atom_sequence, zero = next(inputfile).split()
        assert int(atom_sequence) == sequence_number
        assert int(zero) == 0
        basis = []
        while True:
            line = next(inputfile).rstrip()
            if len(line) == 0:
                break
            shell_label, number_of_primitives, one = line.split()
            assert float(one) == 1.0
            contraction = []
            for i in range(int(number_of_primitives)):
                line = next(inputfile)
                contraction.append(tuple([float(j) for j in line.split()]))
            basis.append((shell_label, contraction))
        return basis

    def _extract_gto(self, inputfile, line):
        if line.startswith('[GTO]'):
            assert hasattr(self, 'atomnos')
            self.gbasis = []
            for sequence in range(self.natom):
                self.gbasis.append(self._extract_one_gto(sequence + 1, inputfile))
            line = next(inputfile)
        return line

    def _extract_mo(self, inputfile, line):
        if line.startswith('[MO]'):
            line = next(inputfile)
        return line

    def extract(self, inputfile, line):
        if self.first_line:
            if not line.startswith('[Molden Format]'):
                raise RuntimeError('file not in Molden format %s' % inputfile)
            self.first_line = False
            return
        line = self._extract_atoms(inputfile, line)
        line = self._extract_gto(inputfile, line)
        line = self._extract_mo(inputfile, line)
        if line.startswith('['):
            raise RuntimeError('unhandled section %s' % line.strip())


if __name__ == '__main__':
    import sys
    import doctest, moldenparser

    if len(sys.argv) == 1:
        doctest.testmod(moldenparser, verbose=False)

    if len(sys.argv) >= 2:
        parser = moldenparser.MOLDEN(sys.argv[1])
        data = parser.parse()

    if len(sys.argv) > 2:
        for i in range(len(sys.argv[2:])):
            if hasattr(data, sys.argv[2 + i]):
                pprint(getattr(data, sys.argv[2 + i]))
