
/* Portions copyright (c) 2006-222 Stanford University and Simbios.
 * Contributors: Pande Group
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "pGM_ReferenceMultipoleForce.h"
#include "SimTKOpenMMRealType.h"
#include "jama_svd.h"
#include <algorithm>
#ifdef _MSC_VER
  #define POCKETFFT_NO_VECTORS
#endif
#include "pocketfft_hdronly.h"

// In case we're using some primitive version of Visual Studio this will
// make sure that erf() and erfc() are defined.
#include "openmm/internal/MSVC_erfc.h"

using std::vector;
using namespace OpenMM;

pGM_ReferenceMultipoleForce::pGM_ReferenceMultipoleForce() :
// larry: need to check these constants
                                                   _nonbondedMethod(IPS),
                                                   _numParticles(0),
                                                   _electric(ONE_4PI_EPS0),
                                                   _dielectric(1.0),
                                                   _mutualInducedDipoleConverged(0),
                                                   _mutualInducedDipoleIterations(0),
                                                   _maximumMutualInducedDipoleIterations(100),
                                                   _mutualInducedDipoleEpsilon(1.0e+50),
                                                   _mutualInducedDipoleTargetEpsilon(1.0e-04),
                                                   _debye(48.033324)
{
    initialize();
}

pGM_ReferenceMultipoleForce::pGM_ReferenceMultipoleForce(NonbondedMethod nonbondedMethod) :
                                                   _nonbondedMethod(nonbondedMethod),
                                                   _numParticles(0),
                                                   _electric(ONE_4PI_EPS0),
                                                   _dielectric(1.0),
                                                   _mutualInducedDipoleConverged(0),
                                                   _mutualInducedDipoleIterations(0),
                                                   _maximumMutualInducedDipoleIterations(100),
                                                   _mutualInducedDipoleEpsilon(1.0e+50),
                                                   _mutualInducedDipoleTargetEpsilon(1.0e-04),
                                                   _debye(48.033324)
{
    initialize();
}

void pGM_ReferenceMultipoleForce::initialize()
{

}

pGM_ReferenceMultipoleForce::NonbondedMethod pGM_ReferenceMultipoleForce::getNonbondedMethod() const
{
    return _nonbondedMethod;
}

void pGM_ReferenceMultipoleForce::setNonbondedMethod(pGM_ReferenceMultipoleForce::NonbondedMethod nonbondedMethod)
{
    _nonbondedMethod = nonbondedMethod;
}

pGM_ReferenceMultipoleForce::PolarizationType pGM_ReferenceMultipoleForce::getPolarizationType() const
{
    return _polarizationType;
}

void pGM_ReferenceMultipoleForce::setPolarizationType(pGM_ReferenceMultipoleForce::PolarizationType polarizationType)
{
    _polarizationType = polarizationType;
}

int pGM_ReferenceMultipoleForce::getMutualInducedDipoleConverged() const
{
    return _mutualInducedDipoleConverged;
}

void pGM_ReferenceMultipoleForce::setMutualInducedDipoleConverged(int mutualInducedDipoleConverged)
{
    _mutualInducedDipoleConverged = mutualInducedDipoleConverged;
}

int pGM_ReferenceMultipoleForce::getMutualInducedDipoleIterations() const
{
    return _mutualInducedDipoleIterations;
}

void pGM_ReferenceMultipoleForce::setMutualInducedDipoleIterations(int mutualInducedDipoleIterations)
{
    _mutualInducedDipoleIterations = mutualInducedDipoleIterations;
}

double pGM_ReferenceMultipoleForce::getMutualInducedDipoleEpsilon() const
{
    return _mutualInducedDipoleEpsilon;
}

void pGM_ReferenceMultipoleForce::setMutualInducedDipoleEpsilon(double mutualInducedDipoleEpsilon)
{
    _mutualInducedDipoleEpsilon = mutualInducedDipoleEpsilon;
}

int pGM_ReferenceMultipoleForce::getMaximumMutualInducedDipoleIterations() const
{
    return _maximumMutualInducedDipoleIterations;
}

void pGM_ReferenceMultipoleForce::setMaximumMutualInducedDipoleIterations(int maximumMutualInducedDipoleIterations)
{
    _maximumMutualInducedDipoleIterations = maximumMutualInducedDipoleIterations;
}

double pGM_ReferenceMultipoleForce::getMutualInducedDipoleTargetEpsilon() const
{
    return _mutualInducedDipoleTargetEpsilon;
}

void pGM_ReferenceMultipoleForce::setMutualInducedDipoleTargetEpsilon(double mutualInducedDipoleTargetEpsilon)
{
    _mutualInducedDipoleTargetEpsilon = mutualInducedDipoleTargetEpsilon;
}


double pGM_ReferenceMultipoleForce::normalizeVec3(Vec3& vectorToNormalize) const
{
    double norm = sqrt(vectorToNormalize.dot(vectorToNormalize));
    if (norm > 0.0) {
        vectorToNormalize *= (1.0/norm);
    }
    return norm;
}

void pGM_ReferenceMultipoleForce::initializeRealOpenMMVector(vector<double>& vectorToInitialize) const
{
    double zero = 0.0;
    vectorToInitialize.resize(_numParticles);
    std::fill(vectorToInitialize.begin(), vectorToInitialize.end(), zero);
}

void pGM_ReferenceMultipoleForce::initializeVec3Vector(vector<Vec3>& vectorToInitialize) const
{
    vectorToInitialize.resize(_numParticles);
    Vec3 zeroVec(0.0, 0.0, 0.0);
    std::fill(vectorToInitialize.begin(), vectorToInitialize.end(), zeroVec);
}

void pGM_ReferenceMultipoleForce::copyVec3Vector(const vector<OpenMM::Vec3>& inputVector, vector<OpenMM::Vec3>& outputVector) const
{
    outputVector.resize(inputVector.size());
    for (unsigned int ii = 0; ii < inputVector.size(); ii++) {
        outputVector[ii] = inputVector[ii];
    }
}

void pGM_ReferenceMultipoleForce::loadParticleData(const vector<Vec3>& particlePositions,
                                                     const vector<double>& charges,
                                                     const vector<double>& dipoles,
                                                     const vector<double>& beta,
                                                     const vector<double>& polarity,
                                                     const vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                                     vector<MultipoleParticleData>& particleData) const
{ 
    vector<int> covalentAtoms;
    particleData.resize(_numParticles);
    int i = 0;
    for (unsigned int ii = 0; ii < _numParticles; ii++) {

        particleData[ii].particleIndex        = ii;

        particleData[ii].position             = particlePositions[ii];
        particleData[ii].charge               = charges[ii];

        particleData[ii].polarity             = polarity[ii];

        covalentAtoms                         = multipoleAtomCovalentInfo[ii][0];
        
        
        Vec3 labDipole;
        for (int jj : covalentAtoms){
            Vec3 vector = particlePositions[jj] - particlePositions[ii];
            double norm = normalizeVec3(vector);
            labDipole[0] = dipoles[i]*vector[0]*norm;
            labDipole[1] = dipoles[i]*vector[1]*norm;
            labDipole[2] = dipoles[i]*vector[2]*norm;
            i++;
        }
        particleData[ii].dipole                = labDipole;
        particleData[ii].beta                  = beta[ii]; 
    }
}

void pGM_ReferenceMultipoleForce::zeroFixedMultipoleFields()
{
    initializeVec3Vector(_fixedMultipoleField);
    initializeVec3Vector(_fixedMultipoleFieldPolar);
}


void pGM_ReferenceMultipoleForce::calculateFixedMultipoleFieldPairIxn(const MultipoleParticleData& particleI,
                                                                        const MultipoleParticleData& particleJ)
{
    if (particleI.particleIndex == particleJ.particleIndex)
        return;

    Vec3 deltaR = particleJ.position - particleI.position;
    double r = sqrt(deltaR.dot(deltaR));

    double betaij = 1 / sqrt(particleI.beta * particleI.beta + particleJ.beta * particleJ.beta);
    double rInv = 1.0 / r;
    double rInv2 = rInv * rInv;
    double rInv3 = rInv * rInv2;


    // Compute the error function term and its exponential derivative
    double erfTerm = erf(betaij * r);
    double expTerm = exp(-betaij * betaij * r * r);
    double erfDerivative = (2 * betaij / sqrt(M_PI)) * expTerm;

    // Second derivative of the error function potential
    double derfrinv = (2 * betaij * betaij * erfDerivative) + 
                      (3 * erfDerivative * rInv2) -
                      (3 * erfTerm * rInv3);

    double prefactor = erfTerm - erfDerivative * r;

    // Field at particle I due to multipoles at particle J
    double dipoleDeltaJ = particleJ.dipole.dot(deltaR);
    Vec3 fieldI = deltaR * (particleJ.charge * prefactor + dipoleDeltaJ * derfrinv) * rInv3 + prefactor * rInv3 * particleJ.dipole;

    unsigned int particleIndexI = particleI.particleIndex;
    _fixedMultipoleField[particleIndexI] -= fieldI;
    _fixedMultipoleFieldPolar[particleIndexI] -= fieldI;

    // Field at particle J due to multipoles at particle I
    double dipoleDeltaI = particleI.dipole.dot(deltaR);
    Vec3 fieldJ = deltaR * (particleI.charge * prefactor + dipoleDeltaI * derfrinv) * rInv3 + prefactor * rInv3 * particleI.dipole;

    unsigned int particleIndexJ = particleJ.particleIndex;
    _fixedMultipoleField[particleIndexJ] += fieldJ;
    _fixedMultipoleFieldPolar[particleIndexJ] += fieldJ;
}

void pGM_ReferenceMultipoleForce::calculateFixedMultipoleField(const vector<MultipoleParticleData>& particleData)
{
    for (unsigned int ii = 0; ii < _numParticles; ii++) {
        for (unsigned int jj = ii + 1; jj < _numParticles; jj++) {
            calculateFixedMultipoleFieldPairIxn(particleData[ii], particleData[jj]);
        }
    }
}

void pGM_ReferenceMultipoleForce::initializeInducedDipoles(vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields)
{

    // initialize inducedDipoles

    _inducedDipole.resize(_numParticles);
    _inducedDipolePolar.resize(_numParticles);

    for (unsigned int ii = 0; ii < _numParticles; ii++) {
        _inducedDipole[ii]       = _fixedMultipoleField[ii];
        _inducedDipolePolar[ii]  = _fixedMultipoleFieldPolar[ii];
    }
}

void pGM_ReferenceMultipoleForce::calculateInducedDipolePairIxn(  unsigned int particleI,
                                                                  unsigned int particleJ,
                                                                  double prefactor,
                                                                  double derfrinv,
                                                                  const Vec3& deltaR,
                                                                  const vector<Vec3>& inducedDipole,
                                                                  vector<Vec3>& field) const
{
    double dDotDelta            = (inducedDipole[particleJ].dot(deltaR));
    field[particleI]           += inducedDipole[particleJ] * prefactor + derfrinv * deltaR*dDotDelta;
    dDotDelta                   = (inducedDipole[particleI].dot(deltaR));
    field[particleJ]           += inducedDipole[particleI] * prefactor + derfrinv * deltaR*dDotDelta;
}

void pGM_ReferenceMultipoleForce::calculateInducedDipolePairIxns(const MultipoleParticleData& particleI,
                                                                   const MultipoleParticleData& particleJ,
                                                                   vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields)
{

   if (particleI.particleIndex == particleJ.particleIndex)
       return;

    Vec3 deltaR   = particleJ.position - particleI.position;
    double r      = sqrt(deltaR.dot(deltaR));
    double betaij = 1 / sqrt(particleI.beta * particleI.beta + particleJ.beta * particleJ.beta);
    double rInv = 1.0 / r;
    double rInv2 = rInv * rInv;
    double rInv3 = rInv * rInv2;


    // Compute the error function term and its exponential derivative
    double erfTerm = erf(betaij * r);
    double expTerm = exp(-betaij * betaij * r * r);
    double erfDerivative = (2 * betaij / sqrt(M_PI)) * expTerm;

    // Second derivative of the error function potential
    double derfrinv = (2 * betaij * betaij * erfDerivative) + 
                      (3 * erfDerivative * rInv2) -
                      (3 * erfTerm * rInv3);

    double prefactor = erfTerm - erfDerivative * r;
    prefactor = prefactor * rInv3;
    derfrinv  = derfrinv  * rInv3;

    for (auto& field : updateInducedDipoleFields) {
        calculateInducedDipolePairIxn(particleI.particleIndex, particleJ.particleIndex, prefactor, derfrinv, deltaR,
                                       *field.inducedDipoles, field.inducedDipoleField);
    }
}

void pGM_ReferenceMultipoleForce::calculateInducedDipoleFields(const vector<MultipoleParticleData>& particleData, vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields) {
    // Initialize the fields to zero.

    Vec3 zeroVec(0.0, 0.0, 0.0);
    for (auto& field : updateInducedDipoleFields)
        std::fill(field.inducedDipoleField.begin(), field.inducedDipoleField.end(), zeroVec);

    // Add fields from all induced dipoles.

    for (unsigned int ii = 0; ii < particleData.size(); ii++)
        for (unsigned int jj = ii; jj < particleData.size(); jj++)
            calculateInducedDipolePairIxns(particleData[ii], particleData[jj], updateInducedDipoleFields);
}



void pGM_ReferenceMultipoleForce::convergeInduceDipolesByDIIS(const vector<MultipoleParticleData>& particleData, vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleField) {
    int numFields = updateInducedDipoleField.size();
    vector<vector<vector<Vec3> > > prevDipoles(numFields);
    vector<vector<Vec3> > prevErrors;
    setMutualInducedDipoleConverged(false);
    int maxPrevious = 20;
    for (int iteration = 0; ; iteration++) {
        // Compute the field from the induced dipoles.

        calculateInducedDipoleFields(particleData, updateInducedDipoleField);

        // Record the current dipoles and the errors in them.

        double maxEpsilon = 0;
        prevErrors.push_back(vector<Vec3>());
        prevErrors.back().resize(_numParticles);
        for (int k = 0; k < numFields; k++) {
            UpdateInducedDipoleFieldStruct& field = updateInducedDipoleField[k];
            prevDipoles[k].push_back(vector<Vec3>());
            prevDipoles[k].back().resize(_numParticles);
            double epsilon = 0;
            for (int i = 0; i < _numParticles; i++) {
                prevDipoles[k].back()[i] = (*field.inducedDipoles)[i];
                Vec3 newDipole = (*field.fixedMultipoleField)[i] + field.inducedDipoleField[i]*particleData[i].polarity;
                Vec3 error = newDipole-(*field.inducedDipoles)[i];
                prevDipoles[k].back()[i] = newDipole;
                if (k == 0)
                    prevErrors.back()[i] = error;
                epsilon += error.dot(error);
            }
            if (epsilon > maxEpsilon)
                maxEpsilon = epsilon;
        }
        maxEpsilon = _debye*sqrt(maxEpsilon/_numParticles);

        // Decide whether to stop or continue iterating.

        if (maxEpsilon < getMutualInducedDipoleTargetEpsilon())
            setMutualInducedDipoleConverged(true);
        if (maxEpsilon < getMutualInducedDipoleTargetEpsilon() || iteration == getMaximumMutualInducedDipoleIterations()) {
            setMutualInducedDipoleEpsilon(maxEpsilon);
            setMutualInducedDipoleIterations(iteration);
            
            return;
        }

        // Select the new dipoles.

        if (prevErrors.size() > maxPrevious) {
            prevErrors.erase(prevErrors.begin());
            for (int k = 0; k < numFields; k++)
                prevDipoles[k].erase(prevDipoles[k].begin());
        }
        int numPrevious = prevErrors.size();
        vector<double> coefficients(numPrevious);
        computeDIISCoefficients(prevErrors, coefficients);
        for (int k = 0; k < numFields; k++) {
            UpdateInducedDipoleFieldStruct& field = updateInducedDipoleField[k];
            for (int i = 0; i < _numParticles; i++) {
                Vec3 dipole(0.0, 0.0, 0.0);
                for (int j = 0; j < numPrevious; j++)
                    dipole += prevDipoles[k][j][i]*coefficients[j];
                (*field.inducedDipoles)[i] = dipole;
            }
        }
    }

}

void pGM_ReferenceMultipoleForce::computeDIISCoefficients(const vector<vector<Vec3> >& prevErrors, vector<double>& coefficients) const {
    int steps = coefficients.size();
    if (steps == 1) {
        coefficients[0] = 1;
        return;
    }

    // Create the DIIS matrix.

    int rank = steps+1;
    Array2D<double> b(rank, rank);
    b[0][0] = 0;
    for (int i = 0; i < steps; i++)
        b[i+1][0] = b[0][i+1] = -1;
    for (int i = 0; i < steps; i++)
        for (int j = i; j < steps; j++) {
            double sum = 0;
            for (int k = 0; k < _numParticles; k++)
                sum += prevErrors[i][k].dot(prevErrors[j][k]);
            b[i+1][j+1] = b[j+1][i+1] = sum;
        }

    // Solve using SVD.  Since the right hand side is (-1, 0, 0, 0, ...), this is simpler than the general case.

    JAMA::SVD<double> svd(b);
    Array2D<double> u, v;
    svd.getU(u);
    svd.getV(v);
    Array1D<double> s;
    svd.getSingularValues(s);
    int effectiveRank = svd.rank();
    for (int i = 1; i < rank; i++) {
        double d = 0;
        for (int j = 0; j < effectiveRank; j++)
            d -= u[0][j]*v[i][j]/s[j];
        coefficients[i-1] = d;
    }
}

void pGM_ReferenceMultipoleForce::calculateInducedDipoles(const vector<MultipoleParticleData>& particleData)
{

    // calculate fixed electric fields

    zeroFixedMultipoleFields();
    calculateFixedMultipoleField(particleData);

    // initialize inducedDipoles
    // if polarization type is 'Direct', then return after initializing; otherwise
    // converge induced dipoles.

    for (unsigned int ii = 0; ii < _numParticles; ii++) {
        _fixedMultipoleField[ii]      *= particleData[ii].polarity;
        _fixedMultipoleFieldPolar[ii] *= particleData[ii].polarity;
    }

    _inducedDipole.resize(_numParticles);
    _inducedDipolePolar.resize(_numParticles);
    vector<UpdateInducedDipoleFieldStruct> updateInducedDipoleField;
    updateInducedDipoleField.push_back(UpdateInducedDipoleFieldStruct(_fixedMultipoleField, _inducedDipole, _ptDipoleD, _ptDipoleFieldGradientD));
    updateInducedDipoleField.push_back(UpdateInducedDipoleFieldStruct(_fixedMultipoleFieldPolar, _inducedDipolePolar, _ptDipoleP, _ptDipoleFieldGradientP));

    initializeInducedDipoles(updateInducedDipoleField);

    //if (getPolarizationType() == pGM_ReferenceMultipoleForce::Direct) {
    //    setMutualInducedDipoleConverged(true);
    //    return;
    //}

    // UpdateInducedDipoleFieldStruct contains induced dipole, fixed multipole fields and fields
    // due to other induced dipoles at each site
    if (getPolarizationType() == pGM_ReferenceMultipoleForce::Mutual){
        convergeInduceDipolesByDIIS(particleData, updateInducedDipoleField); // we also have CG for this
    }else{
        throw OpenMMException("pGM model only supports Mutual induction");
    }
}
double pGM_ReferenceMultipoleForce::calculateElectrostatic(const vector<MultipoleParticleData>& particleData,
                                                             vector<Vec3>& forces)
{
    double energy = 0.0;

    // main loop over particle pairs

    for (unsigned int ii = 0; ii < particleData.size(); ii++) {
        for (unsigned int jj = ii+1; jj < particleData.size(); jj++) {

            energy += calculateElectrostaticPairIxn(particleData[ii], particleData[jj], forces);

        }
    }
    

    return energy;
}

double pGM_ReferenceMultipoleForce::calculateElectrostaticPairIxn(const MultipoleParticleData& particleI,
                                                                        const MultipoleParticleData& particleK,
                                                                        vector<Vec3>& forces) const
{
    unsigned int iIndex = particleI.particleIndex;
    unsigned int kIndex = particleK.particleIndex;

    Vec3 deltaR = particleK.position - particleI.position;
    double r2 = deltaR.dot(deltaR);
    double r = sqrt(r2);

    double qiUindI[3], qiUindJ[3], qiUinpI[3], qiUinpJ[3];


    // The field derivatives at I due to permanent and induced moments on J, and vice-versa.
    // Also, their derivatives w.r.t. R, which are needed for force calculations
    double Vij[9], Vji[9], VjiR[9], VijR[9];
    // The field derivatives at I due to only permanent moments on J, and vice-versa.
    double Vijp[3], Vijd[3], Vjip[3], Vjid[3];
    double rInvVec[7];

    double prefac = (_electric/_dielectric);
    double rInv = 1.0 / r;

    // The rInvVec array is defined such that the ith element is R^-i, with the
    // dieleectric constant folded in, to avoid conversions later.


 

    // Now we compute the (attenuated) Coulomb operator and its derivatives, contracted with
    // permanent moments and induced dipoles.  Note that the coefficient of the permanent force
    // terms is half of the expected value; this is because we compute the interaction of I with
    // the sum of induced and permanent moments on J, as well as the interaction of J with I's
    // permanent and induced moments; doing so double counts the permanent-permanent interaction.
    double energy;
    energy = 1.0;

    // C-C terms (m=0)
  
    return energy;
}


void pGM_ReferenceMultipoleForce::setup(const vector<Vec3>& particlePositions,
                                          const vector<double>& charges,
                                          const vector<double>& dipoles,
                                          const vector<double>& beta,
                                          const vector<double>& polarity,
                                          const vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                          vector<MultipoleParticleData>& particleData)
{


    // load particle parameters into vector of MultipoleParticleData
    // check for inverted chiral centers
    // apply rotation matrix to get lab frame dipole and quadrupoles
    // setup scaling factors
    // get induced dipoles
    // check if induced dipoles converged

    _numParticles = particlePositions.size();
    loadParticleData(particlePositions, charges, dipoles, beta, polarity, multipoleAtomCovalentInfo, particleData);

    calculateInducedDipoles(particleData);

    if (!getMutualInducedDipoleConverged()) {
        std::stringstream message;
        message << "Induced dipoles did not converge: ";
        message << " iterations="      << getMutualInducedDipoleIterations();
        message << " eps="             << getMutualInducedDipoleEpsilon();
        throw OpenMMException(message.str());
    }
}

double pGM_ReferenceMultipoleForce::calculateForceAndEnergy(const vector<Vec3>& particlePositions,
                                                             const vector<double>& charges,
                                                             const vector<double>& dipoles,
                                                             const vector<double>& beta,
                                                             const vector<double>& polarity,
                                                             const vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                                             vector<Vec3>& forces)
{

    // setup, including calculating induced dipoles
    // calculate electrostatic ixns including torques
    // map torques to forces

    vector<MultipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, beta, polarity, 
           multipoleAtomCovalentInfo, particleData);

    double energy = calculateElectrostatic(particleData, forces);

    return energy;
}

void pGM_ReferenceMultipoleForce::calculateInducedDipoles(const vector<Vec3>& particlePositions,
                                                            const vector<double>& charges,
                                                            const vector<double>& dipoles,
                                                            const vector<double>& beta,
                                                            const vector<double>& polarity,
                                                            const vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                                            vector<Vec3>& outputInducedDipoles) {
    // setup, including calculating induced dipoles

    vector<MultipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, beta, polarity, multipoleAtomCovalentInfo, particleData);
    outputInducedDipoles = _inducedDipole;
}




void pGM_ReferenceMultipoleForce::calculateLabFramePermanentDipoles(const vector<Vec3>& particlePositions,
                                                                      const vector<double>& charges,
                                                                      const vector<double>& dipoles,
                                                                      const vector<double>& beta,
                                                                      const vector<double>& polarity,
                                                                      const vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                                                      vector<Vec3>& outputPermanentDipoles) {
    // setup, including calculating permanent dipoles

    vector<MultipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, beta, polarity, multipoleAtomCovalentInfo, particleData);
    outputPermanentDipoles.resize(_numParticles);
    for (int i = 0; i < _numParticles; i++)
        outputPermanentDipoles[i] = particleData[i].dipole;
}

void pGM_ReferenceMultipoleForce::calculateTotalDipoles(const vector<Vec3>& particlePositions,
                                                          const vector<double>& charges,
                                                          const vector<double>& dipoles,
                                                          const vector<double>& beta,
                                                          const vector<double>& polarity,
                                                          const vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                                          vector<Vec3>& outputTotalDipoles) {
    // setup, including calculating permanent dipoles

    vector<MultipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, beta, polarity, multipoleAtomCovalentInfo, particleData);
    outputTotalDipoles.resize(_numParticles);
    for (int i = 0; i < _numParticles; i++)
        for (int j = 0; j < 3; j++)
            outputTotalDipoles[i][j] = particleData[i].dipole[j] + _inducedDipole[i][j];
}

void pGM_ReferenceMultipoleForce::calculateSystemMultipoleMoments(const vector<double>& masses,
                                                                          const vector<Vec3>& particlePositions,
                                                                          const vector<double>& charges,
                                                                          const vector<double>& dipoles,
                                                                          const vector<double>& beta,
                                                                          const vector<double>& polarity,
                                                                          const vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                                                          vector<double>& outputMultipoleMoments)
{

    // setup, including calculating induced dipoles
    // remove center of mass
    // calculate system moments

    vector<MultipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, beta, polarity, multipoleAtomCovalentInfo, particleData);

    double totalMass = 0.0;
    Vec3 centerOfMass = Vec3(0.0, 0.0, 0.0);
    for (unsigned int ii  = 0; ii < _numParticles; ii++) {
        double mass   = masses[ii];
        totalMass    += mass;
        centerOfMass += particleData[ii].position*mass;
    }
    vector<Vec3> localPositions(_numParticles);
    if (totalMass > 0.0) {
        centerOfMass  *= 1.0/totalMass;
    }
    for (unsigned int ii  = 0; ii < _numParticles; ii++) {
        localPositions[ii] = particleData[ii].position - centerOfMass;
    }

    double netchg  = 0.0;

    Vec3 dpl       = Vec3(0.0, 0.0, 0.0);

    double xxqdp   = 0.0;
    double xyqdp   = 0.0;
    double xzqdp   = 0.0;

    double yyqdp   = 0.0;
    double yzqdp   = 0.0;

    double zzqdp   = 0.0;

    for (unsigned int ii  = 0; ii < _numParticles; ii++) {

        double charge         = particleData[ii].charge;
        Vec3 position         = localPositions[ii];
        netchg               += charge;

        Vec3 netDipole        = (particleData[ii].dipole  + _inducedDipole[ii]);

        dpl                  += position*charge + netDipole;

        xxqdp                += position[0]*position[0]*charge + 2.0*position[0]*netDipole[0];
        xyqdp                += position[0]*position[1]*charge + position[0]*netDipole[1] + position[1]*netDipole[0];
        xzqdp                += position[0]*position[2]*charge + position[0]*netDipole[2] + position[2]*netDipole[0];

        yyqdp                += position[1]*position[1]*charge + 2.0*position[1]*netDipole[1];
        yzqdp                += position[1]*position[2]*charge + position[1]*netDipole[2] + position[2]*netDipole[1];

        zzqdp                += position[2]*position[2]*charge + 2.0*position[2]*netDipole[2];

    }

    double debye = 4.80321;

    outputMultipoleMoments[0] = netchg;

    dpl                       *= 10.0*debye;
    outputMultipoleMoments[1]  = dpl[0];
    outputMultipoleMoments[2]  = dpl[1];
    outputMultipoleMoments[3]  = dpl[2];

    debye *= 3.0;
    for (unsigned int ii = 4; ii < 13; ii++) {
        outputMultipoleMoments[ii] *= 100.0*debye;
    }
}



pGM_ReferenceMultipoleForce::UpdateInducedDipoleFieldStruct::UpdateInducedDipoleFieldStruct(vector<OpenMM::Vec3>& inputFixed_E_Field, vector<OpenMM::Vec3>& inputInducedDipoles, vector<vector<Vec3> >& extrapolatedDipoles, vector<vector<double> >& extrapolatedDipoleFieldGradient) :
        fixedMultipoleField(&inputFixed_E_Field), inducedDipoles(&inputInducedDipoles) { 
    inducedDipoleField.resize(fixedMultipoleField->size());
}

