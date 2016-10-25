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

import numpy as np
from collections import namedtuple

from . import logfileparser
from . import utils


Orbital = namedtuple('Orbital', 'symmetry energy spin occupancy mocoeff'.split())


class MOBag(object):
    def __init__(self):
        self.energies = []
        self.coeff = []
        self.syms = []

    def append(self, orbital):
        assert isinstance(orbital, Orbital)
        self.energies.append(orbital.energy)
        self.coeff.append(orbital.mocoeff)
        self.syms.append(orbital.symmetry)

    def __len__(self):
        assert len(self.syms) == len(self.energies) == len(self.coeff)
        return len(self.coeff)


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
        self.cartesian_basis = None

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
            number_of_primitives = int(number_of_primitives)
            assert float(one) == 1.0
            if shell_label == 'SP':
                contraction_s = []
                contraction_p = []
                for i in range(number_of_primitives):
                    line = next(inputfile)
                    fields = [float(j) for j in line.split()]
                    assert len(fields) == 3
                    contraction_s.append((fields[0], fields[1]))
                    contraction_p.append((fields[0], fields[2]))
                basis.append(('S', contraction_s))
                basis.append(('P', contraction_p))
            else:
                contraction = []
                for i in range(number_of_primitives):
                    line = next(inputfile)
                    contraction.append(tuple([float(j) for j in line.split()]))
                    assert len(contraction[-1]) == 2
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

    @staticmethod
    def _extract_one_mo(inputfile, line_symmetry_label, nmo):
        kw_sym, symmetry_label = line_symmetry_label.strip().split('=')
        assert kw_sym == 'Sym'
        kw_ene, mo_energy = next(inputfile).strip().split('=')
        assert kw_ene == 'Ene'
        kw_spin, spin = next(inputfile).strip().split('=')
        assert kw_spin == 'Spin'
        kw_occup, occupation_number = next(inputfile).strip().split('=')
        assert kw_occup == 'Occup'
        if nmo is None:
            mocoeff = []
        else:
            mocoeff = np.zeros([nmo])
        i = 0
        while True:
            try:
                line = next(inputfile).lstrip()
            except StopIteration:
                line = ''
            if line.startswith('Sym') or i == nmo:
                return line, Orbital(symmetry=symmetry_label.strip(), energy=float(mo_energy),
                                     spin=spin.strip(), occupancy=float(occupation_number),
                                     mocoeff=mocoeff)
            ao_number, mo_coefficient = line.split()
            assert int(ao_number) == i + 1
            if nmo is None:
                mocoeff.append(float(mo_coefficient))
            else:
                mocoeff[i] = float(mo_coefficient)
            i += 1

    def _extract_mo(self, inputfile, line):
        if line.startswith('[MO]'):
            nmo = None
            alpha = MOBag()
            beta = MOBag()
            line = next(inputfile)
            while True:
                if line.startswith('[') or len(line.rstrip()) == 0:
                    break
                line, orbital = self._extract_one_mo(inputfile, line, nmo=nmo)
                if orbital.spin == 'Alpha':
                    alpha.append(orbital)
                elif orbital.spin == 'Beta':
                    beta.append(orbital)
                else:
                    raise RuntimeError('unkown spin %s' % orbital.spin)
                if nmo is None:
                    nmo = len(orbital.mocoeff)
            if len(beta) == 0:  # restricted system
                self.moenergies = [np.array(alpha.energies)]
                self.mocoeffs = [np.array(alpha.coeff)]
                self.mosyms = [alpha.syms]
            else:
                self.moenergies = [np.array(alpha.energies), np.array(beta.energies)]
                self.mocoeffs = [np.array(alpha.coeff), np.array(beta.coeff)]
                self.mosyms = [alpha.syms, beta.syms]
            self.set_attribute('nmo', nmo)
        return line

    def _count_nbasis(self):
        dfg_count = {
            True: {'D': 6, 'F': 10, 'G': 15},
            False: {'D': 5, 'F': 7, 'G': 9},
        }
        nbasis = 0
        for aobasis in self.gbasis:
            for shell, contraction in aobasis:
                if shell == 'S':
                    nbasis += 1
                elif shell == 'P':
                    nbasis += 3
                else:
                    cartesian = self.cartesian_basis[shell]
                    nbasis += dfg_count[cartesian][shell]
        self.set_attribute('nbasis', nbasis)

    def _detect_polar_functions(self, inputfile, line):
        self.cartesian_basis = {'D': True, 'F': True, 'G': True}
        if line.startswith('[5D'):
            self.cartesian_basis['D'] = False
            if line.find('10F') > 0:
                pass
            else:
                self.cartesian_basis['F'] = False
            return next(inputfile)
        elif line.startswith('[5D7F]'):
            self.cartesian_basis['D'] = False
            self.cartesian_basis['F'] = False
            return next(inputfile)
        elif line.startswith('[7F]'):
            self.cartesian_basis['F'] = False
            return next(inputfile)
        elif line.startswith('[9G]'):
            self.cartesian_basis['G'] = False
            return next(inputfile)

    def extract(self, inputfile, line):
        if self.first_line:
            if not line.startswith('[Molden Format]'):
                raise RuntimeError('file not in Molden format %s' % inputfile)
            self.first_line = False
            return
        try:
            line = self._extract_atoms(inputfile, line)
            line = self._extract_gto(inputfile, line)
            line = self._extract_mo(inputfile, line)
            line = self._detect_polar_functions(inputfile, line)
            if line is not None and line.startswith('['):
                raise RuntimeError('unhandled section %s' % line.strip())
        except StopIteration:
            pass
        self._count_nbasis()


if __name__ == '__main__':
    import sys
    import doctest, moldenparser
    from pprint import pprint

    if len(sys.argv) == 1:
        doctest.testmod(moldenparser, verbose=False)

    if len(sys.argv) >= 2:
        parser = moldenparser.MOLDEN(sys.argv[1])
        data = parser.parse()

    if len(sys.argv) > 2:
        for i in range(len(sys.argv[2:])):
            if hasattr(data, sys.argv[2 + i]):
                pprint(getattr(data, sys.argv[2 + i]))
