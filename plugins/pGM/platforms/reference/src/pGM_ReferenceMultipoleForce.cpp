
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
                                                   _nonbondedMethod(PME),
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
                                                     const vector<double>& polarity,
                                                     const vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                                     vector<MultipoleParticleData>& particleData) const
{ //larry: here, I changed the dipole vector from 1 dimensional to 2 dimensional to make it look nice
  // another way probably better is to use a iterator for each atom globally instead of locally
  // two dimensional vector takes more meta data
  // And need beta
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
        particleData[ii].beta                  = 1.0; //currently 1.0 for simplicity  
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

double pGM_ReferenceMultipoleForce::calculateElectrostaticPairIxn(const MultipoleParticleData& particleI,
                                                                        const MultipoleParticleData& particleK,
                                                                        const vector<double>& scalingFactors,
                                                                        vector<Vec3>& forces,
                                                                        vector<Vec3>& torque) const
{
    unsigned int iIndex = particleI.particleIndex;
    unsigned int kIndex = particleK.particleIndex;

    Vec3 deltaR = particleK.position - particleI.position;
    double r2 = deltaR.dot(deltaR);
    double r = sqrt(r2);

    double qiUindI[3], qiUindJ[3], qiUinpI[3], qiUinpJ[3];
    for (int ii = 0; ii < 3; ii++) {
        double valIP = 0.0;
        double valID = 0.0;
        double valJP = 0.0;
        double valJD = 0.0;
        for (int jj = 0; jj < 3; jj++) {
            valIP += inducedDipoleRotationMatrix[ii][jj] * _inducedDipolePolar[iIndex][jj];
            valID += inducedDipoleRotationMatrix[ii][jj] * _inducedDipole[iIndex][jj];
            valJP += inducedDipoleRotationMatrix[ii][jj] * _inducedDipolePolar[kIndex][jj];
            valJD += inducedDipoleRotationMatrix[ii][jj] * _inducedDipole[kIndex][jj];
        }
        qiUindI[ii] = valID;
        qiUinpI[ii] = valIP;
        qiUindJ[ii] = valJD;
        qiUinpJ[ii] = valJP;
    }

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
    rInvVec[1] = prefac * rInv;
    for (int i = 2; i < 7; ++i)
        rInvVec[i] = rInvVec[i-1] * rInv;

    double mScale = scalingFactors[M_SCALE];
    double dScale = scalingFactors[D_SCALE];
    double pScale = scalingFactors[P_SCALE];
    double uScale = scalingFactors[U_SCALE];

    double dmp = particleI.dampingFactor*particleK.dampingFactor;
    double a = particleI.thole < particleK.thole ? particleI.thole : particleK.thole;
    double u = r/dmp;
    double au3 = fabs(dmp) > 1.0e-5f ? a*u*u*u : 0.0;
    double expau3 = fabs(dmp) > 1.0e-5f ? exp(-au3) : 0.0;
    double a2u6 = au3*au3;
    double a3u9 = a2u6*au3;
    // Thole damping factors for energies
    double thole_c  = 1.0 - expau3;
    double thole_d0 = 1.0 - expau3*(1.0 + 1.5*au3);
    double thole_d1 = 1.0 - expau3;
    double thole_q0 = 1.0 - expau3*(1.0 + au3 + a2u6);
    double thole_q1 = 1.0 - expau3*(1.0 + au3);
    // Thole damping factors for derivatives
    double dthole_c  = 1.0 - expau3*(1.0 + 1.5*au3);
    double dthole_d0 = 1.0 - expau3*(1.0 + au3 + 1.5*a2u6);
    double dthole_d1 = 1.0 - expau3*(1.0 + au3);
    double dthole_q0 = 1.0 - expau3*(1.0 + au3 + 0.25*a2u6 + 0.75*a3u9);
    double dthole_q1 = 1.0 - expau3*(1.0 + au3 + 0.75*a2u6);

    // Now we compute the (attenuated) Coulomb operator and its derivatives, contracted with
    // permanent moments and induced dipoles.  Note that the coefficient of the permanent force
    // terms is half of the expected value; this is because we compute the interaction of I with
    // the sum of induced and permanent moments on J, as well as the interaction of J with I's
    // permanent and induced moments; doing so double counts the permanent-permanent interaction.
    double ePermCoef, dPermCoef, eUIndCoef, dUIndCoef, eUInpCoef, dUInpCoef;

    // C-C terms (m=0)
    ePermCoef = rInvVec[1]*mScale;
    dPermCoef = -0.5*mScale*rInvVec[2];
    Vij[0]  = ePermCoef*qiQJ[0];
    Vji[0]  = ePermCoef*qiQI[0];
    VijR[0] = dPermCoef*qiQJ[0];
    VjiR[0] = dPermCoef*qiQI[0];

    // C-D and C-Uind terms (m=0)
    ePermCoef = rInvVec[2]*mScale;
    eUIndCoef = rInvVec[2]*pScale*thole_c;
    eUInpCoef = rInvVec[2]*dScale*thole_c;
    dPermCoef = -rInvVec[3]*mScale;
    dUIndCoef = -2.0*rInvVec[3]*pScale*dthole_c;
    dUInpCoef = -2.0*rInvVec[3]*dScale*dthole_c;
    Vij[0]  += -(ePermCoef*qiQJ[1] + eUIndCoef*qiUindJ[0] + eUInpCoef*qiUinpJ[0]);
    Vji[1]   = -(ePermCoef*qiQI[0]);
    VijR[0] += -(dPermCoef*qiQJ[1] + dUIndCoef*qiUindJ[0] + dUInpCoef*qiUinpJ[0]);
    VjiR[1]  = -(dPermCoef*qiQI[0]);
    Vjip[0]  = -(eUInpCoef*qiQI[0]);
    Vjid[0]  = -(eUIndCoef*qiQI[0]);
    // D-C and Uind-C terms (m=0)
    Vij[1]   = ePermCoef*qiQJ[0];
    Vji[0]  += ePermCoef*qiQI[1] + eUIndCoef*qiUindI[0] + eUInpCoef*qiUinpI[0];
    VijR[1]  = dPermCoef*qiQJ[0];
    VjiR[0] += dPermCoef*qiQI[1] + dUIndCoef*qiUindI[0] + dUInpCoef*qiUinpI[0];
    Vijp[0]  = eUInpCoef*qiQJ[0];
    Vijd[0]  = eUIndCoef*qiQJ[0];

    // D-D and D-Uind terms (m=0)
    ePermCoef = -2.0*rInvVec[3]*mScale;
    eUIndCoef = -2.0*rInvVec[3]*pScale*thole_d0;
    eUInpCoef = -2.0*rInvVec[3]*dScale*thole_d0;
    dPermCoef = 3.0*rInvVec[4]*mScale;
    dUIndCoef = 6.0*rInvVec[4]*pScale*dthole_d0;
    dUInpCoef = 6.0*rInvVec[4]*dScale*dthole_d0;
    Vij[1]  += ePermCoef*qiQJ[1] + eUIndCoef*qiUindJ[0] + eUInpCoef*qiUinpJ[0];
    Vji[1]  += ePermCoef*qiQI[1] + eUIndCoef*qiUindI[0] + eUInpCoef*qiUinpI[0];
    VijR[1] += dPermCoef*qiQJ[1] + dUIndCoef*qiUindJ[0] + dUInpCoef*qiUinpJ[0];
    VjiR[1] += dPermCoef*qiQI[1] + dUIndCoef*qiUindI[0] + dUInpCoef*qiUinpI[0];
    Vijp[0] += eUInpCoef*qiQJ[1];
    Vijd[0] += eUIndCoef*qiQJ[1];
    Vjip[0] += eUInpCoef*qiQI[1];
    Vjid[0] += eUIndCoef*qiQI[1];
    // D-D and D-Uind terms (m=1)
    ePermCoef = rInvVec[3]*mScale;
    eUIndCoef = rInvVec[3]*pScale*thole_d1;
    eUInpCoef = rInvVec[3]*dScale*thole_d1;
    dPermCoef = -1.5*rInvVec[4]*mScale;
    dUIndCoef = -3.0*rInvVec[4]*pScale*dthole_d1;
    dUInpCoef = -3.0*rInvVec[4]*dScale*dthole_d1;
    Vij[2]  = ePermCoef*qiQJ[2] + eUIndCoef*qiUindJ[1] + eUInpCoef*qiUinpJ[1];
    Vji[2]  = ePermCoef*qiQI[2] + eUIndCoef*qiUindI[1] + eUInpCoef*qiUinpI[1];
    VijR[2] = dPermCoef*qiQJ[2] + dUIndCoef*qiUindJ[1] + dUInpCoef*qiUinpJ[1];
    VjiR[2] = dPermCoef*qiQI[2] + dUIndCoef*qiUindI[1] + dUInpCoef*qiUinpI[1];
    Vij[3]  = ePermCoef*qiQJ[3] + eUIndCoef*qiUindJ[2] + eUInpCoef*qiUinpJ[2];
    Vji[3]  = ePermCoef*qiQI[3] + eUIndCoef*qiUindI[2] + eUInpCoef*qiUinpI[2];
    VijR[3] = dPermCoef*qiQJ[3] + dUIndCoef*qiUindJ[2] + dUInpCoef*qiUinpJ[2];
    VjiR[3] = dPermCoef*qiQI[3] + dUIndCoef*qiUindI[2] + dUInpCoef*qiUinpI[2];
    Vijp[1] = eUInpCoef*qiQJ[2];
    Vijd[1] = eUIndCoef*qiQJ[2];
    Vjip[1] = eUInpCoef*qiQI[2];
    Vjid[1] = eUIndCoef*qiQI[2];
    Vijp[2] = eUInpCoef*qiQJ[3];
    Vijd[2] = eUIndCoef*qiQJ[3];
    Vjip[2] = eUInpCoef*qiQI[3];
    Vjid[2] = eUIndCoef*qiQI[3];



    // Evaluate the energies, forces and torques due to permanent+induced moments
    // interacting with just the permanent moments.
    double energy = 0.5*(qiQI[0]*Vij[0] + qiQJ[0]*Vji[0]);
    double fIZ = qiQI[0]*VijR[0];
    double fJZ = qiQJ[0]*VjiR[0];
    double EIX = 0.0, EIY = 0.0, EIZ = 0.0, EJX = 0.0, EJY = 0.0, EJZ = 0.0;
    for (int i = 1; i < 9; ++i) {
        energy += 0.5*(qiQI[i]*Vij[i] + qiQJ[i]*Vji[i]);
        fIZ += qiQI[i]*VijR[i];
        fJZ += qiQJ[i]*VjiR[i];
        EIX += qiQIX[i]*Vij[i];
        EIY += qiQIY[i]*Vij[i];
        EIZ += qiQIZ[i]*Vij[i];
        EJX += qiQJX[i]*Vji[i];
        EJY += qiQJY[i]*Vji[i];
        EJZ += qiQJZ[i]*Vji[i];
    }

    // Define the torque intermediates for the induced dipoles. These are simply the induced dipole torque
    // intermediates dotted with the field due to permanent moments only, at each center. We inline the
    // induced dipole torque intermediates here, for simplicity. N.B. There are no torques on the dipoles
    // themselves, so we accumulate the torque intermediates into separate variables to allow them to be
    // used only in the force calculation.
    //
    // The torque about the x axis (needed to obtain the y force on the induced dipoles, below)
    //    qiUindIx[0] = qiQUindI[2];    qiUindIx[1] = 0;    qiUindIx[2] = -qiQUindI[0]
    double iEIX = qiUinpI[2]*Vijp[0] + qiUindI[2]*Vijd[0] - qiUinpI[0]*Vijp[2] - qiUindI[0]*Vijd[2];
    double iEJX = qiUinpJ[2]*Vjip[0] + qiUindJ[2]*Vjid[0] - qiUinpJ[0]*Vjip[2] - qiUindJ[0]*Vjid[2];
    // The torque about the y axis (needed to obtain the x force on the induced dipoles, below)
    //    qiUindIy[0] = -qiQUindI[1];   qiUindIy[1] = qiQUindI[0];    qiUindIy[2] = 0
    double iEIY = qiUinpI[0]*Vijp[1] + qiUindI[0]*Vijd[1] - qiUinpI[1]*Vijp[0] - qiUindI[1]*Vijd[0];
    double iEJY = qiUinpJ[0]*Vjip[1] + qiUindJ[0]*Vjid[1] - qiUinpJ[1]*Vjip[0] - qiUindJ[1]*Vjid[0];

    // Add in the induced-induced terms, if needed.
    if(getPolarizationType() == pGM_ReferenceMultipoleForce::Mutual) {
        // Uind-Uind terms (m=0)
        double eCoef = -4.0*rInvVec[3]*uScale*thole_d0;
        double dCoef = 6.0*rInvVec[4]*uScale*dthole_d0;
        iEIX += eCoef*(qiUinpI[2]*qiUindJ[0] + qiUindI[2]*qiUinpJ[0]);
        iEJX += eCoef*(qiUinpJ[2]*qiUindI[0] + qiUindJ[2]*qiUinpI[0]);
        iEIY -= eCoef*(qiUinpI[1]*qiUindJ[0] + qiUindI[1]*qiUinpJ[0]);
        iEJY -= eCoef*(qiUinpJ[1]*qiUindI[0] + qiUindJ[1]*qiUinpI[0]);
        fIZ += dCoef*(qiUinpI[0]*qiUindJ[0] + qiUindI[0]*qiUinpJ[0]);
        fJZ += dCoef*(qiUinpJ[0]*qiUindI[0] + qiUindJ[0]*qiUinpI[0]);
        // Uind-Uind terms (m=1)
        eCoef = 2.0*rInvVec[3]*uScale*thole_d1;
        dCoef = -3.0*rInvVec[4]*uScale*dthole_d1;
        iEIX -= eCoef*(qiUinpI[0]*qiUindJ[2] + qiUindI[0]*qiUinpJ[2]);
        iEJX -= eCoef*(qiUinpJ[0]*qiUindI[2] + qiUindJ[0]*qiUinpI[2]);
        iEIY += eCoef*(qiUinpI[0]*qiUindJ[1] + qiUindI[0]*qiUinpJ[1]);
        iEJY += eCoef*(qiUinpJ[0]*qiUindI[1] + qiUindJ[0]*qiUinpI[1]);
        fIZ += dCoef*(qiUinpI[1]*qiUindJ[1] + qiUindI[1]*qiUinpJ[1] + qiUinpI[2]*qiUindJ[2] + qiUindI[2]*qiUinpJ[2]);
        fJZ += dCoef*(qiUinpJ[1]*qiUindI[1] + qiUindJ[1]*qiUinpI[1] + qiUinpJ[2]*qiUindI[2] + qiUindJ[2]*qiUinpI[2]);
    }

    // The quasi-internal frame forces and torques.  Note that the induced torque intermediates are
    // used in the force expression, but not in the torques; the induced dipoles are isotropic.
    double qiForce[3] = {rInv*(EIY+EJY+iEIY+iEJY), -rInv*(EIX+EJX+iEIX+iEJX), -(fJZ+fIZ)};
    double qiTorqueI[3] = {-EIX, -EIY, -EIZ};
    double qiTorqueJ[3] = {-EJX, -EJY, -EJZ};

    // Rotate the forces and torques back to the lab frame
    for (int ii = 0; ii < 3; ii++) {
        double forceVal = 0.0;
        double torqueIVal = 0.0;
        double torqueJVal = 0.0;
        for (int jj = 0; jj < 3; jj++) {
            forceVal   += forceRotationMatrix[ii][jj] * qiForce[jj];
            torqueIVal += forceRotationMatrix[ii][jj] * qiTorqueI[jj];
            torqueJVal += forceRotationMatrix[ii][jj] * qiTorqueJ[jj];
        }
        torque[iIndex][ii] += torqueIVal;
        torque[kIndex][ii] += torqueJVal;
        forces[iIndex][ii] -= forceVal;
        forces[kIndex][ii] += forceVal;
    }
    return energy;
}


void pGM_ReferenceMultipoleForce::setup(const vector<Vec3>& particlePositions,
                                          const vector<double>& charges,
                                          const vector<double>& dipoles,
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
    loadParticleData(particlePositions, charges, dipoles, polarity, multipoleAtomCovalentInfo, particleData);

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
                                                             const vector<double>& polarity,
                                                             const vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                                             vector<Vec3>& forces)
{

    // setup, including calculating induced dipoles
    // calculate electrostatic ixns including torques
    // map torques to forces

    vector<MultipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, polarity, 
           multipoleAtomCovalentInfo, particleData);

    double energy = calculateElectrostatic(particleData, forces);

    return energy;
}

void pGM_ReferenceMultipoleForce::calculateInducedDipoles(const vector<Vec3>& particlePositions,
                                                            const vector<double>& charges,
                                                            const vector<double>& dipoles,
                                                            const vector<double>& quadrupoles,
                                                            const vector<double>& tholes,
                                                            const vector<double>& dampingFactors,
                                                            const vector<double>& polarity,
                                                            const vector<int>& axisTypes,
                                                            const vector<int>& multipoleAtomZs,
                                                            const vector<int>& multipoleAtomXs,
                                                            const vector<int>& multipoleAtomYs,
                                                            const vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                                            vector<Vec3>& outputInducedDipoles) {
    // setup, including calculating induced dipoles

    vector<MultipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, polarity, multipoleAtomCovalentInfo, particleData);
    outputInducedDipoles = _inducedDipole;
}




void pGM_ReferenceMultipoleForce::calculateLabFramePermanentDipoles(const vector<Vec3>& particlePositions,
                                                                      const vector<double>& charges,
                                                                      const vector<double>& dipoles,
                                                                      const vector<double>& polarity,
                                                                      const vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                                                      vector<Vec3>& outputPermanentDipoles) {
    // setup, including calculating permanent dipoles

    vector<MultipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, polarity, multipoleAtomCovalentInfo, particleData);
    outputPermanentDipoles.resize(_numParticles);
    for (int i = 0; i < _numParticles; i++)
        outputPermanentDipoles[i] = particleData[i].dipole;
}

void pGM_ReferenceMultipoleForce::calculateTotalDipoles(const vector<Vec3>& particlePositions,
                                                          const vector<double>& charges,
                                                          const vector<double>& dipoles,
                                                          const vector<double>& quadrupoles,
                                                          const vector<double>& tholes,
                                                          const vector<double>& dampingFactors,
                                                          const vector<double>& polarity,
                                                          const vector<int>& axisTypes,
                                                          const vector<int>& multipoleAtomZs,
                                                          const vector<int>& multipoleAtomXs,
                                                          const vector<int>& multipoleAtomYs,
                                                          const vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                                          vector<Vec3>& outputTotalDipoles) {
    // setup, including calculating permanent dipoles

    vector<MultipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, quadrupoles, tholes,
           dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
           multipoleAtomCovalentInfo, particleData);
    outputTotalDipoles.resize(_numParticles);
    for (int i = 0; i < _numParticles; i++)
        for (int j = 0; j < 3; j++)
            outputTotalDipoles[i][j] = particleData[i].dipole[j] + _inducedDipole[i][j];
}

void pGM_ReferenceMultipoleForce::calculate_pGM_SystemMultipoleMoments(const vector<double>& masses,
                                                                          const vector<Vec3>& particlePositions,
                                                                          const vector<double>& charges,
                                                                          const vector<double>& dipoles,
                                                                          const vector<double>& quadrupoles,
                                                                          const vector<double>& tholes,
                                                                          const vector<double>& dampingFactors,
                                                                          const vector<double>& polarity,
                                                                          const vector<int>& axisTypes,
                                                                          const vector<int>& multipoleAtomZs,
                                                                          const vector<int>& multipoleAtomXs,
                                                                          const vector<int>& multipoleAtomYs,
                                                                          const vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                                                          vector<double>& outputMultipoleMoments)
{

    // setup, including calculating induced dipoles
    // remove center of mass
    // calculate system moments

    vector<MultipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, quadrupoles, tholes,
           dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
           multipoleAtomCovalentInfo, particleData);

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

    // convert the quadrupole from traced to traceless form

    outputMultipoleMoments.resize(13);
    double qave                  = (xxqdp + yyqdp + zzqdp)/3.0;
    outputMultipoleMoments[4]    = 0.5*(xxqdp-qave);
    outputMultipoleMoments[5]    = 0.5*xyqdp;
    outputMultipoleMoments[6]    = 0.5*xzqdp;
    outputMultipoleMoments[8]    = 0.5*(yyqdp-qave);
    outputMultipoleMoments[9]    = 0.5*yzqdp;
    outputMultipoleMoments[12]   = 0.5*(zzqdp-qave);

    // add the traceless atomic quadrupoles to total quadrupole

    for (unsigned int ii = 0; ii < _numParticles; ii++) {
        outputMultipoleMoments[4]  += particleData[ii].quadrupole[QXX];
        outputMultipoleMoments[5]  += particleData[ii].quadrupole[QXY];
        outputMultipoleMoments[6]  += particleData[ii].quadrupole[QXZ];
        outputMultipoleMoments[8]  += particleData[ii].quadrupole[QYY];
        outputMultipoleMoments[9]  += particleData[ii].quadrupole[QYZ];
        outputMultipoleMoments[12] += particleData[ii].quadrupole[QZZ];
    }
    outputMultipoleMoments[7]  = outputMultipoleMoments[5];
    outputMultipoleMoments[10] = outputMultipoleMoments[6];
    outputMultipoleMoments[11] = outputMultipoleMoments[9];

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

double pGM_ReferenceMultipoleForce::calculateElectrostaticPotentialForParticleGridPoint(const MultipoleParticleData& particleI, const Vec3& gridPoint) const
{

    Vec3 deltaR = particleI.position - gridPoint;

    getPeriodicDelta(deltaR);

    double r2            = deltaR.dot(deltaR);
    double r             = sqrt(r2);

    double rr1           = 1.0/r;
    double rr2           = rr1*rr1;
    double rr3           = rr1*rr2;
    double potential     = particleI.charge*rr1;

    double scd           = particleI.dipole.dot(deltaR);
    double scu           = _inducedDipole[particleI.particleIndex].dot(deltaR);
    potential           -= (scd + scu)*rr3;

    double rr5           = 3.0*rr3*rr2;
    double scq           = deltaR[0]*(particleI.quadrupole[QXX]*deltaR[0] + particleI.quadrupole[QXY]*deltaR[1] + particleI.quadrupole[QXZ]*deltaR[2]);
          scq           += deltaR[1]*(particleI.quadrupole[QXY]*deltaR[0] + particleI.quadrupole[QYY]*deltaR[1] + particleI.quadrupole[QYZ]*deltaR[2]);
          scq           += deltaR[2]*(particleI.quadrupole[QXZ]*deltaR[0] + particleI.quadrupole[QYZ]*deltaR[1] + particleI.quadrupole[QZZ]*deltaR[2]);
    potential           += scq*rr5;
    return potential;
}

void pGM_ReferenceMultipoleForce::calculateElectrostaticPotential(const vector<Vec3>& particlePositions,
                                                                    const vector<double>& charges,
                                                                    const vector<double>& dipoles,
                                                                    const vector<double>& quadrupoles,
                                                                    const vector<double>& tholes,
                                                                    const vector<double>& dampingFactors,
                                                                    const vector<double>& polarity,
                                                                    const vector<int>& axisTypes,
                                                                    const vector<int>& multipoleAtomZs,
                                                                    const vector<int>& multipoleAtomXs,
                                                                    const vector<int>& multipoleAtomYs,
                                                                    const vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                                                    const vector<Vec3>& grid,
                                                                    vector<double>& potential)
{

    // setup, including calculating induced dipoles
    // initialize potential
    // calculate contribution of each particle to potential at grid point
    // apply prefactor

    vector<MultipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, quadrupoles, tholes,
           dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
           multipoleAtomCovalentInfo, particleData);

    potential.resize(grid.size());
    for (auto& p : potential)
        p = 0.0;

    for (unsigned int ii = 0; ii < _numParticles; ii++) {
        for (unsigned int jj = 0; jj < grid.size(); jj++) {
            potential[jj] += calculateElectrostaticPotentialForParticleGridPoint(particleData[ii], grid[jj]);
        }
    }

    double term = _electric/_dielectric;
    for (auto& p : potential)
        p *= term;
}

pGM_ReferenceMultipoleForce::UpdateInducedDipoleFieldStruct::UpdateInducedDipoleFieldStruct(vector<OpenMM::Vec3>& inputFixed_E_Field, vector<OpenMM::Vec3>& inputInducedDipoles, vector<vector<Vec3> >& extrapolatedDipoles, vector<vector<double> >& extrapolatedDipoleFieldGradient) :
        fixedMultipoleField(&inputFixed_E_Field), inducedDipoles(&inputInducedDipoles), extrapolatedDipoles(&extrapolatedDipoles), extrapolatedDipoleFieldGradient(&extrapolatedDipoleFieldGradient) { 
    inducedDipoleField.resize(fixedMultipoleField->size());
}


const int pGM_ReferencePmeMultipoleForce::pGM_PME_ORDER = 5;

const double pGM_ReferencePmeMultipoleForce::SQRT_PI = 1.77245385091;

pGM_ReferencePmeMultipoleForce::pGM_ReferencePmeMultipoleForce() :
               pGM_ReferenceMultipoleForce(PME),
               _cutoffDistance(1.0), _cutoffDistanceSquared(1.0),
               _pmeGridSize(0), _totalGridSize(0), _alphaEwald(0.0)
{

    _pmeGrid = NULL;
    _pmeGridDimensions = IntVec(-1, -1, -1);
}

pGM_ReferencePmeMultipoleForce::~pGM_ReferencePmeMultipoleForce()
{
    if (_pmeGrid) {
        delete[] _pmeGrid;
    }
};

double pGM_ReferencePmeMultipoleForce::getCutoffDistance() const
{
     return _cutoffDistance;
};

void pGM_ReferencePmeMultipoleForce::setCutoffDistance(double cutoffDistance)
{
     _cutoffDistance        = cutoffDistance;
     _cutoffDistanceSquared = cutoffDistance*cutoffDistance;
};

double pGM_ReferencePmeMultipoleForce::getAlphaEwald() const
{
     return _alphaEwald;
};

void pGM_ReferencePmeMultipoleForce::setAlphaEwald(double alphaEwald)
{
     _alphaEwald = alphaEwald;
};

void pGM_ReferencePmeMultipoleForce::getPmeGridDimensions(vector<int>& pmeGridDimensions) const
{

    pmeGridDimensions.resize(3);

    pmeGridDimensions[0] = _pmeGridDimensions[0];
    pmeGridDimensions[1] = _pmeGridDimensions[1];
    pmeGridDimensions[2] = _pmeGridDimensions[2];
};

void pGM_ReferencePmeMultipoleForce::setPmeGridDimensions(vector<int>& pmeGridDimensions)
{

    if ((pmeGridDimensions[0] == _pmeGridDimensions[0]) &&
        (pmeGridDimensions[1] == _pmeGridDimensions[1]) &&
        (pmeGridDimensions[2] == _pmeGridDimensions[2]))
        return;

    _pmeGridDimensions[0] = pmeGridDimensions[0];
    _pmeGridDimensions[1] = pmeGridDimensions[1];
    _pmeGridDimensions[2] = pmeGridDimensions[2];

    initializeBSplineModuli();
};

void pGM_ReferencePmeMultipoleForce::setPeriodicBoxSize(OpenMM::Vec3* vectors)
{

    if (vectors[0][0] == 0.0 || vectors[1][1] == 0.0 || vectors[2][2] == 0.0) {
        std::stringstream message;
        message << "Box size of zero is invalid.";
        throw OpenMMException(message.str());
    }

    _periodicBoxVectors[0] = vectors[0];
    _periodicBoxVectors[1] = vectors[1];
    _periodicBoxVectors[2] = vectors[2];
    double determinant = vectors[0][0]*vectors[1][1]*vectors[2][2];
    assert(determinant > 0);
    double scale = 1.0/determinant;
    _recipBoxVectors[0] = Vec3(vectors[1][1]*vectors[2][2], 0, 0)*scale;
    _recipBoxVectors[1] = Vec3(-vectors[1][0]*vectors[2][2], vectors[0][0]*vectors[2][2], 0)*scale;
    _recipBoxVectors[2] = Vec3(vectors[1][0]*vectors[2][1]-vectors[1][1]*vectors[2][0], -vectors[0][0]*vectors[2][1], vectors[0][0]*vectors[1][1])*scale;
};

int compareInt2(const int2& v1, const int2& v2)
{
    return v1[1] < v2[1];
}

void pGM_ReferencePmeMultipoleForce::resizePmeArrays()
{

    _totalGridSize = _pmeGridDimensions[0]*_pmeGridDimensions[1]*_pmeGridDimensions[2];
    if (_pmeGridSize < _totalGridSize) {
        if (_pmeGrid) {
            delete[] _pmeGrid;
        }
        _pmeGrid      = new complex<double>[_totalGridSize];
        _pmeGridSize  = _totalGridSize;
    }

    for (unsigned int ii = 0; ii < 3; ii++) {
       _pmeBsplineModuli[ii].resize(_pmeGridDimensions[ii]);
       _thetai[ii].resize(pGM_PME_ORDER*_numParticles);
    }

    _iGrid.resize(_numParticles);
    _phi.resize(20*_numParticles);
    _phid.resize(10*_numParticles);
    _phip.resize(10*_numParticles);
    _phidp.resize(20*_numParticles);
}

void pGM_ReferencePmeMultipoleForce::initializePmeGrid()
{
    if (_pmeGrid == NULL)
        return;

    for (int jj = 0; jj < _totalGridSize; jj++)
        _pmeGrid[jj] = complex<double>(0, 0);
}

void pGM_ReferencePmeMultipoleForce::getPeriodicDelta(Vec3& deltaR) const
{
    deltaR -= _periodicBoxVectors[2]*floor(deltaR[2]*_recipBoxVectors[2][2]+0.5);
    deltaR -= _periodicBoxVectors[1]*floor(deltaR[1]*_recipBoxVectors[1][1]+0.5);
    deltaR -= _periodicBoxVectors[0]*floor(deltaR[0]*_recipBoxVectors[0][0]+0.5);
}



void pGM_ReferencePmeMultipoleForce::initializeBSplineModuli()
{

    // Initialize the b-spline moduli.

    int maxSize = -1;
    for (unsigned int ii = 0; ii < 3; ii++) {
       _pmeBsplineModuli[ii].resize(_pmeGridDimensions[ii]);
        maxSize = maxSize  > _pmeGridDimensions[ii] ? maxSize : _pmeGridDimensions[ii];
    }

    double array[pGM_PME_ORDER];
    double x = 0.0;
    array[0]     = 1.0 - x;
    array[1]     = x;
    for (int k = 2; k < pGM_PME_ORDER; k++) {
        double denom = 1.0/k;
        array[k] = x*array[k-1]*denom;
        for (int i = 1; i < k; i++) {
            array[k-i] = ((x+i)*array[k-i-1] + ((k-i+1)-x)*array[k-i])*denom;
        }
        array[0] = (1.0-x)*array[0]*denom;
    }

    vector<double> bsarray(maxSize+1, 0.0);
    for (int i = 2; i <= pGM_PME_ORDER+1; i++) {
        bsarray[i] = array[i-2];
    }
    for (int dim = 0; dim < 3; dim++) {

        int size = _pmeGridDimensions[dim];

        // get the modulus of the discrete Fourier transform

        double factor = 2.0*M_PI/size;
        for (int i = 0; i < size; i++) {
            double sum1 = 0.0;
            double sum2 = 0.0;
            for (int j = 1; j <= size; j++) {
                double arg = factor*i*(j-1);
                sum1 += bsarray[j]*cos(arg);
                sum2 += bsarray[j]*sin(arg);
            }
            _pmeBsplineModuli[dim][i] = (sum1*sum1 + sum2*sum2);
        }

        // fix for exponential Euler spline interpolation failure

        double eps = 1.0e-7;
        if (_pmeBsplineModuli[dim][0] < eps) {
            _pmeBsplineModuli[dim][0] = 0.5*_pmeBsplineModuli[dim][1];
        }
        for (int i = 1; i < size-1; i++) {
            if (_pmeBsplineModuli[dim][i] < eps) {
                _pmeBsplineModuli[dim][i] = 0.5*(_pmeBsplineModuli[dim][i-1]+_pmeBsplineModuli[dim][i+1]);
            }
        }
        if (_pmeBsplineModuli[dim][size-1] < eps) {
            _pmeBsplineModuli[dim][size-1] = 0.5*_pmeBsplineModuli[dim][size-2];
        }

        // compute and apply the optimal zeta coefficient

        int jcut = 50;
        for (int i = 1; i <= size; i++) {
            int k = i - 1;
            if (i > size/2)
                k = k - size;
            double zeta;
            if (k == 0)
                zeta = 1.0;
            else {
                double sum1 = 1.0;
                double sum2 = 1.0;
                factor = M_PI*k/size;
                for (int j = 1; j <= jcut; j++) {
                    double arg = factor/(factor+M_PI*j);
                    sum1 = sum1 + pow(arg,   pGM_PME_ORDER);
                    sum2 = sum2 + pow(arg, 2*pGM_PME_ORDER);
                }
                for (int j = 1; j <= jcut; j++) {
                    double arg  = factor/(factor-M_PI*j);
                    sum1 += pow(arg,   pGM_PME_ORDER);
                    sum2 += pow(arg, 2*pGM_PME_ORDER);
                }
                zeta = sum2/sum1;
            }
            _pmeBsplineModuli[dim][i-1] = _pmeBsplineModuli[dim][i-1]*(zeta*zeta);
        }
    }
}

void pGM_ReferencePmeMultipoleForce::calculateFixedMultipoleFieldPairIxn(const MultipoleParticleData& particleI,
                                                                           const MultipoleParticleData& particleJ,
                                                                           double dscale, double pscale)
{

    unsigned int iIndex    = particleI.particleIndex;
    unsigned int jIndex    = particleJ.particleIndex;

    // compute the real space portion of the Ewald summation

    if (particleI.particleIndex == particleJ.particleIndex)
        return;

    Vec3 deltaR = particleJ.position - particleI.position;
    getPeriodicDelta(deltaR);
    double r2 = deltaR.dot(deltaR);

    if (r2 > _cutoffDistanceSquared)
        return;

    double r           = sqrt(r2);

    // calculate the error function damping terms

    double ralpha      = _alphaEwald*r;

    double bn0         = erfc(ralpha)/r;
    double alsq2       = 2.0*_alphaEwald*_alphaEwald;
    double alsq2n      = 1.0/(SQRT_PI*_alphaEwald);
    double exp2a       = exp(-(ralpha*ralpha));
    alsq2n            *= alsq2;
    double bn1         = (bn0+alsq2n*exp2a)/r2;

    alsq2n            *= alsq2;
    double bn2         = (3.0*bn1+alsq2n*exp2a)/r2;

    alsq2n            *= alsq2;
    double bn3         = (5.0*bn2+alsq2n*exp2a)/r2;

    // compute the error function scaled and unscaled terms

    Vec3 dampedDInverseDistances;
    Vec3 dampedPInverseDistances;
    getDampedInverseDistances(particleI, particleJ, dscale, pscale, r, dampedDInverseDistances, dampedPInverseDistances);

    double drr3        = dampedDInverseDistances[0];
    double drr5        = dampedDInverseDistances[1];
    double drr7        = dampedDInverseDistances[2];

    double prr3        = dampedPInverseDistances[0];
    double prr5        = dampedPInverseDistances[1];
    double prr7        = dampedPInverseDistances[2];

    double dir         = particleI.dipole.dot(deltaR);

    Vec3 qxI           = Vec3(particleI.quadrupole[QXX], particleI.quadrupole[QXY], particleI.quadrupole[QXZ]);
    Vec3 qyI           = Vec3(particleI.quadrupole[QXY], particleI.quadrupole[QYY], particleI.quadrupole[QYZ]);
    Vec3 qzI           = Vec3(particleI.quadrupole[QXZ], particleI.quadrupole[QYZ], particleI.quadrupole[QZZ]);

    Vec3 qi            = Vec3(qxI.dot(deltaR), qyI.dot(deltaR), qzI.dot(deltaR));
    double qir         = qi.dot(deltaR);

    double djr         = particleJ.dipole.dot(deltaR);

    Vec3 qxJ           = Vec3(particleJ.quadrupole[QXX], particleJ.quadrupole[QXY], particleJ.quadrupole[QXZ]);
    Vec3 qyJ           = Vec3(particleJ.quadrupole[QXY], particleJ.quadrupole[QYY], particleJ.quadrupole[QYZ]);
    Vec3 qzJ           = Vec3(particleJ.quadrupole[QXZ], particleJ.quadrupole[QYZ], particleJ.quadrupole[QZZ]);

    Vec3 qj            = Vec3(qxJ.dot(deltaR), qyJ.dot(deltaR), qzJ.dot(deltaR));
    double qjr         = qj.dot(deltaR);

    Vec3 fim           = qj*(2.0*bn2)  - particleJ.dipole*bn1  - deltaR*(bn1*particleJ.charge - bn2*djr+bn3*qjr);
    Vec3 fjm           = qi*(-2.0*bn2)  - particleI.dipole*bn1  + deltaR*(bn1*particleI.charge + bn2*dir+bn3*qir);

    Vec3 fid           = qj*(2.0*drr5) - particleJ.dipole*drr3 - deltaR*(drr3*particleJ.charge - drr5*djr+drr7*qjr);
    Vec3 fjd           = qi*(-2.0*drr5) - particleI.dipole*drr3 + deltaR*(drr3*particleI.charge + drr5*dir+drr7*qir);

    Vec3 fip           = qj*(2.0*prr5) - particleJ.dipole*prr3 - deltaR*(prr3*particleJ.charge - prr5*djr+prr7*qjr);
    Vec3 fjp           = qi*(-2.0*prr5) - particleI.dipole*prr3 + deltaR*(prr3*particleI.charge + prr5*dir+prr7*qir);

    // increment the field at each site due to this interaction


    _fixedMultipoleField[iIndex]      += fim - fid;
    _fixedMultipoleField[jIndex]      += fjm - fjd;

    _fixedMultipoleFieldPolar[iIndex] += fim - fip;
    _fixedMultipoleFieldPolar[jIndex] += fjm - fjp;
}

void pGM_ReferencePmeMultipoleForce::calculateFixedMultipoleField(const vector<MultipoleParticleData>& particleData)
{

    // first calculate reciprocal space fixed multipole fields

    resizePmeArrays();
    computeBsplines(particleData);
    initializePmeGrid();
    spreadFixedMultipolesOntoGrid(particleData);
    vector<size_t> shape = {(size_t) _pmeGridDimensions[0], (size_t) _pmeGridDimensions[1], (size_t) _pmeGridDimensions[2]};
    vector<size_t> axes = {0, 1, 2};
    vector<ptrdiff_t> stride = {(ptrdiff_t) (_pmeGridDimensions[1]*_pmeGridDimensions[2]*sizeof(complex<double>)),
                                (ptrdiff_t) (_pmeGridDimensions[2]*sizeof(complex<double>)),
                                (ptrdiff_t) sizeof(complex<double>)};
    pocketfft::c2c(shape, stride, stride, axes, true, _pmeGrid, _pmeGrid, 1.0, 0);
    performReciprocalConvolution();
    pocketfft::c2c(shape, stride, stride, axes, false, _pmeGrid, _pmeGrid, 1.0, 0);
    computeFixedPotentialFromGrid();
    recordFixedMultipoleField();

    // include self-energy portion of the multipole field
    // and initialize _fixedMultipoleFieldPolar to _fixedMultipoleField

    double term = (4.0/3.0)*(_alphaEwald*_alphaEwald*_alphaEwald)/SQRT_PI;
    for (unsigned int jj = 0; jj < _numParticles; jj++) {
        Vec3 selfEnergy = particleData[jj].dipole*term;
        _fixedMultipoleField[jj] += selfEnergy;
        _fixedMultipoleFieldPolar[jj] = _fixedMultipoleField[jj];
    }

    // include direct space fixed multipole fields

    this->pGM_ReferenceMultipoleForce::calculateFixedMultipoleField(particleData);
}

#define ARRAY(x,y) array[(x)-1+((y)-1)*pGM_PME_ORDER]

/**
 * This is called from computeBsplines().  It calculates the spline coefficients for a single atom along a single axis.
 */
void pGM_ReferencePmeMultipoleForce::computeBSplinePoint(vector<double4>& thetai, double w)
{

    double array[pGM_PME_ORDER*pGM_PME_ORDER];

    // initialization to get to 2nd order recursion

    ARRAY(2,2) = w;
    ARRAY(2,1) = 1.0 - w;

    // perform one pass to get to 3rd order recursion

    ARRAY(3,3) = 0.5 * w * ARRAY(2,2);
    ARRAY(3,2) = 0.5 * ((1.0+w)*ARRAY(2,1)+(2.0-w)*ARRAY(2,2));
    ARRAY(3,1) = 0.5 * (1.0-w) * ARRAY(2,1);

    // compute standard B-spline recursion to desired order

    for (int i = 4; i <= pGM_PME_ORDER; i++) {
        int k = i - 1;
        double denom = 1.0 / k;
        ARRAY(i,i) = denom * w * ARRAY(k,k);
        for (int j = 1; j <= i-2; j++)
            ARRAY(i,i-j) = denom * ((w+j)*ARRAY(k,i-j-1)+(i-j-w)*ARRAY(k,i-j));
        ARRAY(i,1) = denom * (1.0-w) * ARRAY(k,1);
    }

    // get coefficients for the B-spline first derivative

    int k = pGM_PME_ORDER - 1;
    ARRAY(k,pGM_PME_ORDER) = ARRAY(k,pGM_PME_ORDER-1);
    for (int i = pGM_PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);

    // get coefficients for the B-spline second derivative

    k = pGM_PME_ORDER - 2;
    ARRAY(k,pGM_PME_ORDER-1) = ARRAY(k,pGM_PME_ORDER-2);
    for (int i = pGM_PME_ORDER-2; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,pGM_PME_ORDER) = ARRAY(k,pGM_PME_ORDER-1);
    for (int i = pGM_PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);

    // get coefficients for the B-spline third derivative

    k = pGM_PME_ORDER - 3;
    ARRAY(k,pGM_PME_ORDER-2) = ARRAY(k,pGM_PME_ORDER-3);
    for (int i = pGM_PME_ORDER-3; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,pGM_PME_ORDER-1) = ARRAY(k,pGM_PME_ORDER-2);
    for (int i = pGM_PME_ORDER-2; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,pGM_PME_ORDER) = ARRAY(k,pGM_PME_ORDER-1);
    for (int i = pGM_PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);

    // copy coefficients from temporary to permanent storage

    for (int i = 1; i <= pGM_PME_ORDER; i++)
        thetai[i-1] = double4(ARRAY(pGM_PME_ORDER,i), ARRAY(pGM_PME_ORDER-1,i), ARRAY(pGM_PME_ORDER-2,i), ARRAY(pGM_PME_ORDER-3,i));
}

/**
 * Compute b-spline coefficients.
 */
void pGM_ReferencePmeMultipoleForce::computeBsplines(const vector<MultipoleParticleData>& particleData)
{
    //  get the B-spline coefficients for each multipole site

    for (unsigned int ii = 0; ii < _numParticles; ii++) {
        Vec3 position  = particleData[ii].position;
        getPeriodicDelta(position);
        IntVec igrid;
        for (unsigned int jj = 0; jj < 3; jj++) {

            double w  = position[0]*_recipBoxVectors[0][jj]+position[1]*_recipBoxVectors[1][jj]+position[2]*_recipBoxVectors[2][jj];
            double fr = _pmeGridDimensions[jj]*(w-(int)(w+0.5)+0.5);
            int ifr   = static_cast<int>(floor(fr));
            w         = fr - ifr;
            igrid[jj] = ifr - pGM_PME_ORDER + 1;
            igrid[jj] += igrid[jj] < 0 ? _pmeGridDimensions[jj] : 0;
            vector<double4> thetaiTemp(pGM_PME_ORDER);
            computeBSplinePoint(thetaiTemp, w);
            for (unsigned int kk = 0; kk < pGM_PME_ORDER; kk++)
                _thetai[jj][ii*pGM_PME_ORDER+kk] = thetaiTemp[kk];
        }

        // Record the grid point.

        _iGrid[ii]               = igrid;
    }
}

void pGM_ReferencePmeMultipoleForce::transformMultipolesToFractionalCoordinates(const vector<MultipoleParticleData>& particleData) {
    // Build matrices for transforming the dipoles and quadrupoles.

    Vec3 a[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            a[j][i] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];
    int index1[] = {0, 0, 0, 1, 1, 2};
    int index2[] = {0, 1, 2, 1, 2, 2};
    double b[6][6];
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            b[i][j] = a[index1[i]][index1[j]]*a[index2[i]][index2[j]];
            if (index1[i] != index2[i])
                b[i][j] += a[index1[i]][index2[j]]*a[index2[i]][index1[j]];
        }
    }

    // Transform the multipoles.

    _transformed.resize(particleData.size());
    double quadScale[] = {1, 2, 2, 1, 2, 1};
    for (int i = 0; i < (int) particleData.size(); i++) {
        _transformed[i].charge = particleData[i].charge;
        _transformed[i].dipole = Vec3();
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                _transformed[i].dipole[j] += a[j][k]*particleData[i].dipole[k];
    }
}

void pGM_ReferencePmeMultipoleForce::transformPotentialToCartesianCoordinates(const vector<double>& fphi, vector<double>& cphi) const {
    // Build a matrix for transforming the potential.

    Vec3 a[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            a[i][j] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];
    int index1[] = {0, 1, 2, 0, 0, 1};
    int index2[] = {0, 1, 2, 1, 2, 2};
    double b[6][6];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 6; j++) {
            b[i][j] = a[index1[i]][index1[j]]*a[index2[i]][index2[j]];
            if (index1[j] != index2[j])
                b[i][j] *= 2;
        }
    }
    for (int i = 3; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            b[i][j] = a[index1[i]][index1[j]]*a[index2[i]][index2[j]];
            if (index1[j] != index2[j])
                b[i][j] += a[index1[i]][index2[j]]*a[index2[i]][index1[j]];
        }
    }

    // Transform the potential.

    for (int i = 0; i < _numParticles; i++) {
        cphi[10*i] = fphi[20*i];
        cphi[10*i+1] = a[0][0]*fphi[20*i+1] + a[0][1]*fphi[20*i+2] + a[0][2]*fphi[20*i+3];
        cphi[10*i+2] = a[1][0]*fphi[20*i+1] + a[1][1]*fphi[20*i+2] + a[1][2]*fphi[20*i+3];
        cphi[10*i+3] = a[2][0]*fphi[20*i+1] + a[2][1]*fphi[20*i+2] + a[2][2]*fphi[20*i+3];
        for (int j = 0; j < 6; j++) {
            cphi[10*i+4+j] = 0;
            for (int k = 0; k < 6; k++)
                cphi[10*i+4+j] += b[j][k]*fphi[20*i+4+k];
        }
    }
}

void pGM_ReferencePmeMultipoleForce::spreadFixedMultipolesOntoGrid(const vector<MultipoleParticleData>& particleData)
{

    transformMultipolesToFractionalCoordinates(particleData);

    // Clear the grid.

    for (int gridIndex = 0; gridIndex < _totalGridSize; gridIndex++)
        _pmeGrid[gridIndex] = complex<double>(0, 0);

    // Loop over atoms and spread them on the grid.

    for (int atomIndex = 0; atomIndex < _numParticles; atomIndex++) {
        double atomCharge = _transformed[atomIndex].charge;
        Vec3 atomDipole = Vec3(_transformed[atomIndex].dipole[0],
                               _transformed[atomIndex].dipole[1],
                               _transformed[atomIndex].dipole[2]);


        IntVec& gridPoint = _iGrid[atomIndex];
        for (int ix = 0; ix < pGM_PME_ORDER; ix++) {
            int x = (gridPoint[0]+ix) % _pmeGridDimensions[0];
            double4 t = _thetai[0][atomIndex*pGM_PME_ORDER+ix];
            for (int iy = 0; iy < pGM_PME_ORDER; iy++) {
                int y = (gridPoint[1]+iy) % _pmeGridDimensions[1];
                double4 u = _thetai[1][atomIndex*pGM_PME_ORDER+iy];
                double term0 = atomCharge*t[0]*u[0] + atomDipole[1]*t[0]*u[1] + atomQuadrupoleYY*t[0]*u[2] + atomDipole[0]*t[1]*u[0] + atomQuadrupoleXY*t[1]*u[1] + atomQuadrupoleXX*t[2]*u[0];
                double term1 = atomDipole[2]*t[0]*u[0] + atomQuadrupoleYZ*t[0]*u[1] + atomQuadrupoleXZ*t[1]*u[0];
                double term2 = atomQuadrupoleZZ*t[0]*u[0];
                for (int iz = 0; iz < pGM_PME_ORDER; iz++) {
                    int z = (gridPoint[2]+iz) % _pmeGridDimensions[2];
                    double4 v = _thetai[2][atomIndex*pGM_PME_ORDER+iz];
                    complex<double>& gridValue = _pmeGrid[x*_pmeGridDimensions[1]*_pmeGridDimensions[2]+y*_pmeGridDimensions[2]+z];
                    gridValue += term0*v[0] + term1*v[1] + term2*v[2];
                }
            }
        }
    }
}

void pGM_ReferencePmeMultipoleForce::performReciprocalConvolution()
{

    double expFactor   = (M_PI*M_PI)/(_alphaEwald*_alphaEwald);
    double scaleFactor = 1.0/(M_PI*_periodicBoxVectors[0][0]*_periodicBoxVectors[1][1]*_periodicBoxVectors[2][2]);

    for (int index = 0; index < _totalGridSize; index++)
    {
        int kx = index/(_pmeGridDimensions[1]*_pmeGridDimensions[2]);
        int remainder = index-kx*_pmeGridDimensions[1]*_pmeGridDimensions[2];
        int ky = remainder/_pmeGridDimensions[2];
        int kz = remainder-ky*_pmeGridDimensions[2];

        if (kx == 0 && ky == 0 && kz == 0) {
            _pmeGrid[index] = complex<double>(0, 0);
            continue;
        }

        int mx = (kx < (_pmeGridDimensions[0]+1)/2) ? kx : (kx-_pmeGridDimensions[0]);
        int my = (ky < (_pmeGridDimensions[1]+1)/2) ? ky : (ky-_pmeGridDimensions[1]);
        int mz = (kz < (_pmeGridDimensions[2]+1)/2) ? kz : (kz-_pmeGridDimensions[2]);

        double mhx = mx*_recipBoxVectors[0][0];
        double mhy = mx*_recipBoxVectors[1][0]+my*_recipBoxVectors[1][1];
        double mhz = mx*_recipBoxVectors[2][0]+my*_recipBoxVectors[2][1]+mz*_recipBoxVectors[2][2];

        double bx = _pmeBsplineModuli[0][kx];
        double by = _pmeBsplineModuli[1][ky];
        double bz = _pmeBsplineModuli[2][kz];

        double m2 = mhx*mhx+mhy*mhy+mhz*mhz;
        double denom = m2*bx*by*bz;
        double eterm = scaleFactor*exp(-expFactor*m2)/denom;

        _pmeGrid[index] *= eterm;
    }
}

void pGM_ReferencePmeMultipoleForce::computeFixedPotentialFromGrid()
{
    // extract the permanent multipole field at each site

    for (int m = 0; m < _numParticles; m++) {
        IntVec gridPoint = _iGrid[m];
        double tuv000 = 0.0;
        double tuv001 = 0.0;
        double tuv010 = 0.0;
        double tuv100 = 0.0;
        double tuv200 = 0.0;
        double tuv020 = 0.0;
        double tuv002 = 0.0;
        double tuv110 = 0.0;
        double tuv101 = 0.0;
        double tuv011 = 0.0;
        double tuv300 = 0.0;
        double tuv030 = 0.0;
        double tuv003 = 0.0;
        double tuv210 = 0.0;
        double tuv201 = 0.0;
        double tuv120 = 0.0;
        double tuv021 = 0.0;
        double tuv102 = 0.0;
        double tuv012 = 0.0;
        double tuv111 = 0.0;
        for (int iz = 0; iz < pGM_PME_ORDER; iz++) {
            int k = gridPoint[2]+iz-(gridPoint[2]+iz >= _pmeGridDimensions[2] ? _pmeGridDimensions[2] : 0);
            double4 v = _thetai[2][m*pGM_PME_ORDER+iz];
            double tu00 = 0.0;
            double tu10 = 0.0;
            double tu01 = 0.0;
            double tu20 = 0.0;
            double tu11 = 0.0;
            double tu02 = 0.0;
            double tu30 = 0.0;
            double tu21 = 0.0;
            double tu12 = 0.0;
            double tu03 = 0.0;
            for (int iy = 0; iy < pGM_PME_ORDER; iy++) {
                int j = gridPoint[1]+iy-(gridPoint[1]+iy >= _pmeGridDimensions[1] ? _pmeGridDimensions[1] : 0);
                double4 u = _thetai[1][m*pGM_PME_ORDER+iy];
                double4 t = double4(0.0, 0.0, 0.0, 0.0);
                for (int ix = 0; ix < pGM_PME_ORDER; ix++) {
                    int i = gridPoint[0]+ix-(gridPoint[0]+ix >= _pmeGridDimensions[0] ? _pmeGridDimensions[0] : 0);
                    int gridIndex = i*_pmeGridDimensions[1]*_pmeGridDimensions[2] + j*_pmeGridDimensions[2] + k;
                    double tq = _pmeGrid[gridIndex].real();
                    double4 tadd = _thetai[0][m*pGM_PME_ORDER+ix];
                    t[0] += tq*tadd[0];
                    t[1] += tq*tadd[1];
                    t[2] += tq*tadd[2];
                    t[3] += tq*tadd[3];
                }
                tu00 += t[0]*u[0];
                tu10 += t[1]*u[0];
                tu01 += t[0]*u[1];
                tu20 += t[2]*u[0];
                tu11 += t[1]*u[1];
                tu02 += t[0]*u[2];
                tu30 += t[3]*u[0];
                tu21 += t[2]*u[1];
                tu12 += t[1]*u[2];
                tu03 += t[0]*u[3];
            }
            tuv000 += tu00*v[0];
            tuv100 += tu10*v[0];
            tuv010 += tu01*v[0];
            tuv001 += tu00*v[1];
            tuv200 += tu20*v[0];
            tuv020 += tu02*v[0];
            tuv002 += tu00*v[2];
            tuv110 += tu11*v[0];
            tuv101 += tu10*v[1];
            tuv011 += tu01*v[1];
            tuv300 += tu30*v[0];
            tuv030 += tu03*v[0];
            tuv003 += tu00*v[3];
            tuv210 += tu21*v[0];
            tuv201 += tu20*v[1];
            tuv120 += tu12*v[0];
            tuv021 += tu02*v[1];
            tuv102 += tu10*v[2];
            tuv012 += tu01*v[2];
            tuv111 += tu11*v[1];
        }
        _phi[20*m] = tuv000;
        _phi[20*m+1] = tuv100;
        _phi[20*m+2] = tuv010;
        _phi[20*m+3] = tuv001;
        _phi[20*m+4] = tuv200;
        _phi[20*m+5] = tuv020;
        _phi[20*m+6] = tuv002;
        _phi[20*m+7] = tuv110;
        _phi[20*m+8] = tuv101;
        _phi[20*m+9] = tuv011;
        _phi[20*m+10] = tuv300;
        _phi[20*m+11] = tuv030;
        _phi[20*m+12] = tuv003;
        _phi[20*m+13] = tuv210;
        _phi[20*m+14] = tuv201;
        _phi[20*m+15] = tuv120;
        _phi[20*m+16] = tuv021;
        _phi[20*m+17] = tuv102;
        _phi[20*m+18] = tuv012;
        _phi[20*m+19] = tuv111;
    }
}

void pGM_ReferencePmeMultipoleForce::spreadInducedDipolesOnGrid(const vector<Vec3>& inputInducedDipole,
                                                                  const vector<Vec3>& inputInducedDipolePolar) {
    // Create the matrix to convert from Cartesian to fractional coordinates.

    Vec3 cartToFrac[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            cartToFrac[j][i] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];

    // Clear the grid.

    for (int gridIndex = 0; gridIndex < _totalGridSize; gridIndex++)
        _pmeGrid[gridIndex] = complex<double>(0, 0);

    // Loop over atoms and spread them on the grid.

    for (int atomIndex = 0; atomIndex < _numParticles; atomIndex++) {
        Vec3 inducedDipole = Vec3(inputInducedDipole[atomIndex][0]*cartToFrac[0][0] + inputInducedDipole[atomIndex][1]*cartToFrac[0][1] + inputInducedDipole[atomIndex][2]*cartToFrac[0][2],
                                  inputInducedDipole[atomIndex][0]*cartToFrac[1][0] + inputInducedDipole[atomIndex][1]*cartToFrac[1][1] + inputInducedDipole[atomIndex][2]*cartToFrac[1][2],
                                  inputInducedDipole[atomIndex][0]*cartToFrac[2][0] + inputInducedDipole[atomIndex][1]*cartToFrac[2][1] + inputInducedDipole[atomIndex][2]*cartToFrac[2][2]);
        Vec3 inducedDipolePolar = Vec3(inputInducedDipolePolar[atomIndex][0]*cartToFrac[0][0] + inputInducedDipolePolar[atomIndex][1]*cartToFrac[0][1] + inputInducedDipolePolar[atomIndex][2]*cartToFrac[0][2],
                                       inputInducedDipolePolar[atomIndex][0]*cartToFrac[1][0] + inputInducedDipolePolar[atomIndex][1]*cartToFrac[1][1] + inputInducedDipolePolar[atomIndex][2]*cartToFrac[1][2],
                                       inputInducedDipolePolar[atomIndex][0]*cartToFrac[2][0] + inputInducedDipolePolar[atomIndex][1]*cartToFrac[2][1] + inputInducedDipolePolar[atomIndex][2]*cartToFrac[2][2]);
        IntVec& gridPoint = _iGrid[atomIndex];
        for (int ix = 0; ix < pGM_PME_ORDER; ix++) {
            int x = (gridPoint[0]+ix) % _pmeGridDimensions[0];
            double4 t = _thetai[0][atomIndex*pGM_PME_ORDER+ix];
            for (int iy = 0; iy < pGM_PME_ORDER; iy++) {
                int y = (gridPoint[1]+iy) % _pmeGridDimensions[1];
                double4 u = _thetai[1][atomIndex*pGM_PME_ORDER+iy];
                double term01 = inducedDipole[1]*t[0]*u[1] + inducedDipole[0]*t[1]*u[0];
                double term11 = inducedDipole[2]*t[0]*u[0];
                double term02 = inducedDipolePolar[1]*t[0]*u[1] + inducedDipolePolar[0]*t[1]*u[0];
                double term12 = inducedDipolePolar[2]*t[0]*u[0];
                for (int iz = 0; iz < pGM_PME_ORDER; iz++) {
                    int z = (gridPoint[2]+iz) % _pmeGridDimensions[2];
                    double4 v = _thetai[2][atomIndex*pGM_PME_ORDER+iz];
                    complex<double>& gridValue = _pmeGrid[x*_pmeGridDimensions[1]*_pmeGridDimensions[2]+y*_pmeGridDimensions[2]+z];
                    gridValue += complex<double>(term01*v[0] + term11*v[1], term02*v[0] + term12*v[1]);
                }
            }
        }
    }
}

void pGM_ReferencePmeMultipoleForce::computeInducedPotentialFromGrid()
{
    // extract the induced dipole field at each site

    for (int m = 0; m < _numParticles; m++) {
        IntVec gridPoint = _iGrid[m];
        double tuv100_1 = 0.0;
        double tuv010_1 = 0.0;
        double tuv001_1 = 0.0;
        double tuv200_1 = 0.0;
        double tuv020_1 = 0.0;
        double tuv002_1 = 0.0;
        double tuv110_1 = 0.0;
        double tuv101_1 = 0.0;
        double tuv011_1 = 0.0;
        double tuv100_2 = 0.0;
        double tuv010_2 = 0.0;
        double tuv001_2 = 0.0;
        double tuv200_2 = 0.0;
        double tuv020_2 = 0.0;
        double tuv002_2 = 0.0;
        double tuv110_2 = 0.0;
        double tuv101_2 = 0.0;
        double tuv011_2 = 0.0;
        double tuv000 = 0.0;
        double tuv001 = 0.0;
        double tuv010 = 0.0;
        double tuv100 = 0.0;
        double tuv200 = 0.0;
        double tuv020 = 0.0;
        double tuv002 = 0.0;
        double tuv110 = 0.0;
        double tuv101 = 0.0;
        double tuv011 = 0.0;
        double tuv300 = 0.0;
        double tuv030 = 0.0;
        double tuv003 = 0.0;
        double tuv210 = 0.0;
        double tuv201 = 0.0;
        double tuv120 = 0.0;
        double tuv021 = 0.0;
        double tuv102 = 0.0;
        double tuv012 = 0.0;
        double tuv111 = 0.0;
        for (int iz = 0; iz < pGM_PME_ORDER; iz++) {
            int k = gridPoint[2]+iz-(gridPoint[2]+iz >= _pmeGridDimensions[2] ? _pmeGridDimensions[2] : 0);
            double4 v = _thetai[2][m*pGM_PME_ORDER+iz];
            double tu00_1 = 0.0;
            double tu01_1 = 0.0;
            double tu10_1 = 0.0;
            double tu20_1 = 0.0;
            double tu11_1 = 0.0;
            double tu02_1 = 0.0;
            double tu00_2 = 0.0;
            double tu01_2 = 0.0;
            double tu10_2 = 0.0;
            double tu20_2 = 0.0;
            double tu11_2 = 0.0;
            double tu02_2 = 0.0;
            double tu00 = 0.0;
            double tu10 = 0.0;
            double tu01 = 0.0;
            double tu20 = 0.0;
            double tu11 = 0.0;
            double tu02 = 0.0;
            double tu30 = 0.0;
            double tu21 = 0.0;
            double tu12 = 0.0;
            double tu03 = 0.0;
            for (int iy = 0; iy < pGM_PME_ORDER; iy++) {
                int j = gridPoint[1]+iy-(gridPoint[1]+iy >= _pmeGridDimensions[1] ? _pmeGridDimensions[1] : 0);
                double4 u = _thetai[1][m*pGM_PME_ORDER+iy];
                double t0_1 = 0.0;
                double t1_1 = 0.0;
                double t2_1 = 0.0;
                double t0_2 = 0.0;
                double t1_2 = 0.0;
                double t2_2 = 0.0;
                double t3 = 0.0;
                for (int ix = 0; ix < pGM_PME_ORDER; ix++) {
                    int i = gridPoint[0]+ix-(gridPoint[0]+ix >= _pmeGridDimensions[0] ? _pmeGridDimensions[0] : 0);
                    int gridIndex = i*_pmeGridDimensions[1]*_pmeGridDimensions[2] + j*_pmeGridDimensions[2] + k;
                    complex<double> tq = _pmeGrid[gridIndex];
                    double4 tadd = _thetai[0][m*pGM_PME_ORDER+ix];
                    t0_1 += tq.real()*tadd[0];
                    t1_1 += tq.real()*tadd[1];
                    t2_1 += tq.real()*tadd[2];
                    t0_2 += tq.imag()*tadd[0];
                    t1_2 += tq.imag()*tadd[1];
                    t2_2 += tq.imag()*tadd[2];
                    t3 += (tq.real()+tq.imag())*tadd[3];
                }
                tu00_1 += t0_1*u[0];
                tu10_1 += t1_1*u[0];
                tu01_1 += t0_1*u[1];
                tu20_1 += t2_1*u[0];
                tu11_1 += t1_1*u[1];
                tu02_1 += t0_1*u[2];
                tu00_2 += t0_2*u[0];
                tu10_2 += t1_2*u[0];
                tu01_2 += t0_2*u[1];
                tu20_2 += t2_2*u[0];
                tu11_2 += t1_2*u[1];
                tu02_2 += t0_2*u[2];
                double t0 = t0_1 + t0_2;
                double t1 = t1_1 + t1_2;
                double t2 = t2_1 + t2_2;
                tu00 += t0*u[0];
                tu10 += t1*u[0];
                tu01 += t0*u[1];
                tu20 += t2*u[0];
                tu11 += t1*u[1];
                tu02 += t0*u[2];
                tu30 += t3*u[0];
                tu21 += t2*u[1];
                tu12 += t1*u[2];
                tu03 += t0*u[3];
            }
            tuv100_1 += tu10_1*v[0];
            tuv010_1 += tu01_1*v[0];
            tuv001_1 += tu00_1*v[1];
            tuv200_1 += tu20_1*v[0];
            tuv020_1 += tu02_1*v[0];
            tuv002_1 += tu00_1*v[2];
            tuv110_1 += tu11_1*v[0];
            tuv101_1 += tu10_1*v[1];
            tuv011_1 += tu01_1*v[1];
            tuv100_2 += tu10_2*v[0];
            tuv010_2 += tu01_2*v[0];
            tuv001_2 += tu00_2*v[1];
            tuv200_2 += tu20_2*v[0];
            tuv020_2 += tu02_2*v[0];
            tuv002_2 += tu00_2*v[2];
            tuv110_2 += tu11_2*v[0];
            tuv101_2 += tu10_2*v[1];
            tuv011_2 += tu01_2*v[1];
            tuv000 += tu00*v[0];
            tuv100 += tu10*v[0];
            tuv010 += tu01*v[0];
            tuv001 += tu00*v[1];
            tuv200 += tu20*v[0];
            tuv020 += tu02*v[0];
            tuv002 += tu00*v[2];
            tuv110 += tu11*v[0];
            tuv101 += tu10*v[1];
            tuv011 += tu01*v[1];
            tuv300 += tu30*v[0];
            tuv030 += tu03*v[0];
            tuv003 += tu00*v[3];
            tuv210 += tu21*v[0];
            tuv201 += tu20*v[1];
            tuv120 += tu12*v[0];
            tuv021 += tu02*v[1];
            tuv102 += tu10*v[2];
            tuv012 += tu01*v[2];
            tuv111 += tu11*v[1];
        }
        _phid[10*m]   = 0.0;
        _phid[10*m+1] = tuv100_1;
        _phid[10*m+2] = tuv010_1;
        _phid[10*m+3] = tuv001_1;
        _phid[10*m+4] = tuv200_1;
        _phid[10*m+5] = tuv020_1;
        _phid[10*m+6] = tuv002_1;
        _phid[10*m+7] = tuv110_1;
        _phid[10*m+8] = tuv101_1;
        _phid[10*m+9] = tuv011_1;

        _phip[10*m]   = 0.0;
        _phip[10*m+1] = tuv100_2;
        _phip[10*m+2] = tuv010_2;
        _phip[10*m+3] = tuv001_2;
        _phip[10*m+4] = tuv200_2;
        _phip[10*m+5] = tuv020_2;
        _phip[10*m+6] = tuv002_2;
        _phip[10*m+7] = tuv110_2;
        _phip[10*m+8] = tuv101_2;
        _phip[10*m+9] = tuv011_2;

        _phidp[20*m] = tuv000;
        _phidp[20*m+1] = tuv100;
        _phidp[20*m+2] = tuv010;
        _phidp[20*m+3] = tuv001;
        _phidp[20*m+4] = tuv200;
        _phidp[20*m+5] = tuv020;
        _phidp[20*m+6] = tuv002;
        _phidp[20*m+7] = tuv110;
        _phidp[20*m+8] = tuv101;
        _phidp[20*m+9] = tuv011;
        _phidp[20*m+10] = tuv300;
        _phidp[20*m+11] = tuv030;
        _phidp[20*m+12] = tuv003;
        _phidp[20*m+13] = tuv210;
        _phidp[20*m+14] = tuv201;
        _phidp[20*m+15] = tuv120;
        _phidp[20*m+16] = tuv021;
        _phidp[20*m+17] = tuv102;
        _phidp[20*m+18] = tuv012;
        _phidp[20*m+19] = tuv111;
    }
}

double pGM_ReferencePmeMultipoleForce::computeReciprocalSpaceFixedMultipoleForceAndEnergy(const vector<MultipoleParticleData>& particleData,
                                                                                            vector<Vec3>& forces, vector<Vec3>& torques) const
{
    double multipole[10];
    const int deriv1[] = {1, 4, 7, 8, 10, 15, 17, 13, 14, 19};
    const int deriv2[] = {2, 7, 5, 9, 13, 11, 18, 15, 19, 16};
    const int deriv3[] = {3, 8, 9, 6, 14, 16, 12, 19, 17, 18};
    vector<double> cphi(10*_numParticles);
    transformPotentialToCartesianCoordinates(_phi, cphi);
    Vec3 fracToCart[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            fracToCart[i][j] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];
    double energy = 0.0;
    for (int i = 0; i < _numParticles; i++) {

        // Compute the torque.

        multipole[0] = particleData[i].charge;

        multipole[1] =  particleData[i].dipole[0];
        multipole[2] =  particleData[i].dipole[1];
        multipole[3] =  particleData[i].dipole[2];


        const double* phi = &cphi[10*i];
        torques[i][0] += _electric*(multipole[3]*phi[2] - multipole[2]*phi[3]
                      + 2.0*(multipole[6]-multipole[5])*phi[9]
                      + multipole[8]*phi[7] + multipole[9]*phi[5]
                      - multipole[7]*phi[8] - multipole[9]*phi[6]);

        torques[i][1] += _electric*(multipole[1]*phi[3] - multipole[3]*phi[1]
                      + 2.0*(multipole[4]-multipole[6])*phi[8]
                      + multipole[7]*phi[9] + multipole[8]*phi[6]
                      - multipole[8]*phi[4] - multipole[9]*phi[7]);

        torques[i][2] += _electric*(multipole[2]*phi[1] - multipole[1]*phi[2]
                      + 2.0*(multipole[5]-multipole[4])*phi[7]
                      + multipole[7]*phi[4] + multipole[9]*phi[8]
                      - multipole[7]*phi[5] - multipole[8]*phi[9]);

        // Compute the force and energy.

        multipole[1] = _transformed[i].dipole[0];
        multipole[2] = _transformed[i].dipole[1];
        multipole[3] = _transformed[i].dipole[2];

        Vec3 f = Vec3(0.0, 0.0, 0.0);
        for (int k = 0; k < 10; k++) {
            energy += multipole[k]*_phi[20*i+k];
            f[0]   += multipole[k]*_phi[20*i+deriv1[k]];
            f[1]   += multipole[k]*_phi[20*i+deriv2[k]];
            f[2]   += multipole[k]*_phi[20*i+deriv3[k]];
        }
        f              *= (_electric);
        forces[i]      -= Vec3(f[0]*fracToCart[0][0] + f[1]*fracToCart[0][1] + f[2]*fracToCart[0][2],
                               f[0]*fracToCart[1][0] + f[1]*fracToCart[1][1] + f[2]*fracToCart[1][2],
                               f[0]*fracToCart[2][0] + f[1]*fracToCart[2][1] + f[2]*fracToCart[2][2]);

    }
    return (0.5*_electric*energy);
}

/**
 * Compute the forces due to the reciprocal space PME calculation for induced dipoles.
 */
double pGM_ReferencePmeMultipoleForce::computeReciprocalSpaceInducedDipoleForceAndEnergy(pGM_ReferenceMultipoleForce::PolarizationType polarizationType,
                                                                                           const vector<MultipoleParticleData>& particleData,
                                                                                           vector<Vec3>& forces, vector<Vec3>& torques) const
{
    double multipole[10];
    double inducedDipole[3];
    double inducedDipolePolar[3];
    double scales[3];
    const int deriv1[] = {1, 4, 7, 8, 10, 15, 17, 13, 14, 19};
    const int deriv2[] = {2, 7, 5, 9, 13, 11, 18, 15, 19, 16};
    const int deriv3[] = {3, 8, 9, 6, 14, 16, 12, 19, 17, 18};
    vector<double> cphi(10*_numParticles);
    transformPotentialToCartesianCoordinates(_phidp, cphi);
    Vec3 cartToFrac[3], fracToCart[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            cartToFrac[j][i] = fracToCart[i][j] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];
    double energy = 0.0;
    for (int i = 0; i < _numParticles; i++) {

        // Compute the torque.

        unsigned int iIndex = particleData[i].particleIndex;

        multipole[0] = particleData[i].charge;

        multipole[1] = particleData[i].dipole[0];
        multipole[2] = particleData[i].dipole[1];
        multipole[3] = particleData[i].dipole[2];



        const double* phi = &cphi[10*i];
        torques[iIndex][0] += 0.5*_electric*(multipole[3]*phi[2] - multipole[2]*phi[3]
                      + 2.0*(multipole[6]-multipole[5])*phi[9]
                      + multipole[8]*phi[7] + multipole[9]*phi[5]
                      - multipole[7]*phi[8] - multipole[9]*phi[6]);

        torques[iIndex][1] += 0.5*_electric*(multipole[1]*phi[3] - multipole[3]*phi[1]
                      + 2.0*(multipole[4]-multipole[6])*phi[8]
                      + multipole[7]*phi[9] + multipole[8]*phi[6]
                      - multipole[8]*phi[4] - multipole[9]*phi[7]);

        torques[iIndex][2] += 0.5*_electric*(multipole[2]*phi[1] - multipole[1]*phi[2]
                      + 2.0*(multipole[5]-multipole[4])*phi[7]
                      + multipole[7]*phi[4] + multipole[9]*phi[8]
                      - multipole[7]*phi[5] - multipole[8]*phi[9]);

        // Compute the force and energy.

        multipole[1] = _transformed[i].dipole[0];
        multipole[2] = _transformed[i].dipole[1];
        multipole[3] = _transformed[i].dipole[2];


        inducedDipole[0] = _inducedDipole[i][0]*cartToFrac[0][0] + _inducedDipole[i][1]*cartToFrac[0][1] + _inducedDipole[i][2]*cartToFrac[0][2];
        inducedDipole[1] = _inducedDipole[i][0]*cartToFrac[1][0] + _inducedDipole[i][1]*cartToFrac[1][1] + _inducedDipole[i][2]*cartToFrac[1][2];
        inducedDipole[2] = _inducedDipole[i][0]*cartToFrac[2][0] + _inducedDipole[i][1]*cartToFrac[2][1] + _inducedDipole[i][2]*cartToFrac[2][2];
        inducedDipolePolar[0] = _inducedDipolePolar[i][0]*cartToFrac[0][0] + _inducedDipolePolar[i][1]*cartToFrac[0][1] + _inducedDipolePolar[i][2]*cartToFrac[0][2];
        inducedDipolePolar[1] = _inducedDipolePolar[i][0]*cartToFrac[1][0] + _inducedDipolePolar[i][1]*cartToFrac[1][1] + _inducedDipolePolar[i][2]*cartToFrac[1][2];
        inducedDipolePolar[2] = _inducedDipolePolar[i][0]*cartToFrac[2][0] + _inducedDipolePolar[i][1]*cartToFrac[2][1] + _inducedDipolePolar[i][2]*cartToFrac[2][2];

        energy += (inducedDipole[0]+inducedDipolePolar[0])*_phi[20*i+1];
        energy += (inducedDipole[1]+inducedDipolePolar[1])*_phi[20*i+2];
        energy += (inducedDipole[2]+inducedDipolePolar[2])*_phi[20*i+3];

        Vec3 f = Vec3(0.0, 0.0, 0.0);

        for (int k = 0; k < 3; k++) {

            int j1 = deriv1[k+1];
            int j2 = deriv2[k+1];
            int j3 = deriv3[k+1];

            f[0] += (inducedDipole[k]+inducedDipolePolar[k])*_phi[20*i+j1];
            f[1] += (inducedDipole[k]+inducedDipolePolar[k])*_phi[20*i+j2];
            f[2] += (inducedDipole[k]+inducedDipolePolar[k])*_phi[20*i+j3];
 
            if (polarizationType == pGM_ReferenceMultipoleForce::Mutual) {
                f[0] += (inducedDipole[k]*_phip[10*i+j1] + inducedDipolePolar[k]*_phid[10*i+j1]);
                f[1] += (inducedDipole[k]*_phip[10*i+j2] + inducedDipolePolar[k]*_phid[10*i+j2]);
                f[2] += (inducedDipole[k]*_phip[10*i+j3] + inducedDipolePolar[k]*_phid[10*i+j3]);
            }
        }

        for (int k = 0; k < 10; k++) {
            f[0] += multipole[k]*_phidp[20*i+deriv1[k]];
            f[1] += multipole[k]*_phidp[20*i+deriv2[k]];
            f[2] += multipole[k]*_phidp[20*i+deriv3[k]];
        }

        f              *= (0.5*_electric);
        forces[iIndex] -= Vec3(f[0]*fracToCart[0][0] + f[1]*fracToCart[0][1] + f[2]*fracToCart[0][2],
                               f[0]*fracToCart[1][0] + f[1]*fracToCart[1][1] + f[2]*fracToCart[1][2],
                               f[0]*fracToCart[2][0] + f[1]*fracToCart[2][1] + f[2]*fracToCart[2][2]);
    }
    return (0.25*_electric*energy);
}

void pGM_ReferencePmeMultipoleForce::recordFixedMultipoleField()
{
    Vec3 fracToCart[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            fracToCart[i][j] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];
    for (int i = 0; i < _numParticles; i++) {
        _fixedMultipoleField[i][0] = -(_phi[20*i+1]*fracToCart[0][0] + _phi[20*i+2]*fracToCart[0][1] + _phi[20*i+3]*fracToCart[0][2]);
        _fixedMultipoleField[i][1] = -(_phi[20*i+1]*fracToCart[1][0] + _phi[20*i+2]*fracToCart[1][1] + _phi[20*i+3]*fracToCart[1][2]);
        _fixedMultipoleField[i][2] = -(_phi[20*i+1]*fracToCart[2][0] + _phi[20*i+2]*fracToCart[2][1] + _phi[20*i+3]*fracToCart[2][2]);
    }
}

void pGM_ReferencePmeMultipoleForce::initializeInducedDipoles(vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields)
{

    this->pGM_ReferenceMultipoleForce::initializeInducedDipoles(updateInducedDipoleFields);
    calculateReciprocalSpaceInducedDipoleField(updateInducedDipoleFields);
}

void pGM_ReferencePmeMultipoleForce::recordInducedDipoleField(vector<Vec3>& field, vector<Vec3>& fieldPolar)
{
    Vec3 fracToCart[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            fracToCart[i][j] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];
    for (int i = 0; i < _numParticles; i++) {

        field[i][0] -= _phid[10*i+1]*fracToCart[0][0] + _phid[10*i+2]*fracToCart[0][1] + _phid[10*i+3]*fracToCart[0][2];
        field[i][1] -= _phid[10*i+1]*fracToCart[1][0] + _phid[10*i+2]*fracToCart[1][1] + _phid[10*i+3]*fracToCart[1][2];
        field[i][2] -= _phid[10*i+1]*fracToCart[2][0] + _phid[10*i+2]*fracToCart[2][1] + _phid[10*i+3]*fracToCart[2][2];

        fieldPolar[i][0] -= _phip[10*i+1]*fracToCart[0][0] + _phip[10*i+2]*fracToCart[0][1] + _phip[10*i+3]*fracToCart[0][2];
        fieldPolar[i][1] -= _phip[10*i+1]*fracToCart[1][0] + _phip[10*i+2]*fracToCart[1][1] + _phip[10*i+3]*fracToCart[1][2];
        fieldPolar[i][2] -= _phip[10*i+1]*fracToCart[2][0] + _phip[10*i+2]*fracToCart[2][1] + _phip[10*i+3]*fracToCart[2][2];
    }
}

void pGM_ReferencePmeMultipoleForce::calculateReciprocalSpaceInducedDipoleField(vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields)
{
    // Perform PME for the induced dipoles.

    initializePmeGrid();
    spreadInducedDipolesOnGrid(*updateInducedDipoleFields[0].inducedDipoles, *updateInducedDipoleFields[1].inducedDipoles);
    vector<size_t> shape = {(size_t) _pmeGridDimensions[0], (size_t) _pmeGridDimensions[1], (size_t) _pmeGridDimensions[2]};
    vector<size_t> axes = {0, 1, 2};
    vector<ptrdiff_t> stride = {(ptrdiff_t) (_pmeGridDimensions[1]*_pmeGridDimensions[2]*sizeof(complex<double>)),
                                (ptrdiff_t) (_pmeGridDimensions[2]*sizeof(complex<double>)),
                                (ptrdiff_t) sizeof(complex<double>)};
    pocketfft::c2c(shape, stride, stride, axes, true, _pmeGrid, _pmeGrid, 1.0, 0);
    performReciprocalConvolution();
    pocketfft::c2c(shape, stride, stride, axes, false, _pmeGrid, _pmeGrid, 1.0, 0);
    computeInducedPotentialFromGrid();
    recordInducedDipoleField(updateInducedDipoleFields[0].inducedDipoleField, updateInducedDipoleFields[1].inducedDipoleField);
}

void pGM_ReferencePmeMultipoleForce::calculateInducedDipoleFields(const vector<MultipoleParticleData>& particleData,
                                                                     vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields)
{
    // Initialize the fields to zero.

    Vec3 zeroVec(0.0, 0.0, 0.0);
    for (auto& field : updateInducedDipoleFields)
        std::fill(field.inducedDipoleField.begin(), field.inducedDipoleField.end(), zeroVec);

    // Add fields from direct space interactions.

    for (unsigned int ii = 0; ii < particleData.size(); ii++) {
        for (unsigned int jj = ii + 1; jj < particleData.size(); jj++) {
            calculateDirectInducedDipolePairIxns(particleData[ii], particleData[jj], updateInducedDipoleFields);
        }
    }

    // reciprocal space ixns

    calculateReciprocalSpaceInducedDipoleField(updateInducedDipoleFields);


    // self ixn

    double term = (4.0/3.0)*(_alphaEwald*_alphaEwald*_alphaEwald)/SQRT_PI;
    for (auto& field : updateInducedDipoleFields) {
        vector<Vec3>& inducedDipoles = *field.inducedDipoles;
        vector<Vec3>& inducedDipoleField = field.inducedDipoleField;
        for (unsigned int jj = 0; jj < particleData.size(); jj++) {
            inducedDipoleField[jj] += inducedDipoles[jj]*term;
        }
    }
}

void pGM_ReferencePmeMultipoleForce::calculateDirectInducedDipolePairIxn(unsigned int iIndex, unsigned int jIndex,
                                                                           double preFactor1, double preFactor2,
                                                                           const Vec3& delta,
                                                                           const vector<Vec3>& inducedDipole,
                                                                           vector<Vec3>& field) const
{

    // field at i due induced dipole at j

    double dur  = inducedDipole[jIndex].dot(delta);
    field[iIndex]  += delta*(dur*preFactor2) + inducedDipole[jIndex]*preFactor1;

    // field at j due induced dipole at i

               dur  = inducedDipole[iIndex].dot(delta);
    field[jIndex]  += delta*(dur*preFactor2) + inducedDipole[iIndex]*preFactor1;
}

void pGM_ReferencePmeMultipoleForce::calculateDirectInducedDipolePairIxns(const MultipoleParticleData& particleI,
                                                                            const MultipoleParticleData& particleJ,
                                                                            vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields)
{

    // compute the real space portion of the Ewald summation

    double uscale = 1.0;
    Vec3 deltaR = particleJ.position - particleI.position;

    // periodic boundary conditions

    getPeriodicDelta(deltaR);
    double r2 = deltaR.dot(deltaR);

    if (r2 > _cutoffDistanceSquared)
        return;

    double r           = sqrt(r2);

    // calculate the error function damping terms

    double ralpha      = _alphaEwald*r;

    double bn0         = erfc(ralpha)/r;
    double alsq2       = 2.0*_alphaEwald*_alphaEwald;
    double alsq2n      = 1.0/(SQRT_PI*_alphaEwald);
    double exp2a       = exp(-(ralpha*ralpha));
    alsq2n            *= alsq2;
    double bn1         = (bn0+alsq2n*exp2a)/r2;

    alsq2n            *= alsq2;
    double bn2         = (3.0*bn1+alsq2n*exp2a)/r2;

    alsq2n            *= alsq2;
    double bn3         = (5.0*bn2+alsq2n*exp2a)/r2;

    // compute the error function scaled and unscaled terms

    double scale3      = 1.0;
    double scale5      = 1.0;
    double scale7      = 1.0;
    double damp        = particleI.dampingFactor*particleJ.dampingFactor;
    if (damp != 0.0) {

        double ratio = (r/damp);
               ratio = ratio*ratio*ratio;
        double pgamma = particleI.thole < particleJ.thole ? particleI.thole : particleJ.thole;
               damp   = -pgamma*ratio;

        if (damp > -50.0) {
            double expdamp = expf(damp);
            scale3        = 1.0 - expdamp;
            scale5        = 1.0 - expdamp*(1.0-damp);
            scale7        = 1.0 - (1.0 - damp + (0.6*damp*damp))*expdamp;
        }
    }
    double dsc3        = uscale*scale3;
    double dsc5        = uscale*scale5;
    double dsc7        = uscale*scale7;

    double r3          = (r*r2);
    double r5          = (r3*r2);
    double r7          = (r5*r2);
    double rr3         = (1.0-dsc3)/r3;
    double rr5         = 3.0*(1.0-dsc5)/r5;
    double rr7         = 15.0*(1.0-dsc7)/r7;

    double preFactor1  = rr3 - bn1;
    double preFactor2  = bn2 - rr5;
    double preFactor3  = bn3 - rr7;

    for (auto& field : updateInducedDipoleFields) {
        calculateDirectInducedDipolePairIxn(particleI.particleIndex, particleJ.particleIndex, preFactor1, preFactor2, deltaR,
                                            *field.inducedDipoles, field.inducedDipoleField);
        if (getPolarizationType() == pGM_ReferenceMultipoleForce::Extrapolated) {
            // Compute and store the field gradient for later use.
            double dx = deltaR[0];
            double dy = deltaR[1];
            double dz = deltaR[2];

            OpenMM::Vec3 &dipolesI = (*field.inducedDipoles)[particleI.particleIndex];
            double xDipole = dipolesI[0];
            double yDipole = dipolesI[1];
            double zDipole = dipolesI[2];
            double muDotR = xDipole*dx + yDipole*dy + zDipole*dz;
            double Exx = muDotR*dx*dx*preFactor3 - (2.0*xDipole*dx + muDotR)*preFactor2;
            double Eyy = muDotR*dy*dy*preFactor3 - (2.0*yDipole*dy + muDotR)*preFactor2;
            double Ezz = muDotR*dz*dz*preFactor3 - (2.0*zDipole*dz + muDotR)*preFactor2;
            double Exy = muDotR*dx*dy*preFactor3 - (xDipole*dy + yDipole*dx)*preFactor2;
            double Exz = muDotR*dx*dz*preFactor3 - (xDipole*dz + zDipole*dx)*preFactor2;
            double Eyz = muDotR*dy*dz*preFactor3 - (yDipole*dz + zDipole*dy)*preFactor2;

            field.inducedDipoleFieldGradient[particleJ.particleIndex][0] -= Exx;
            field.inducedDipoleFieldGradient[particleJ.particleIndex][1] -= Eyy;
            field.inducedDipoleFieldGradient[particleJ.particleIndex][2] -= Ezz;
            field.inducedDipoleFieldGradient[particleJ.particleIndex][3] -= Exy;
            field.inducedDipoleFieldGradient[particleJ.particleIndex][4] -= Exz;
            field.inducedDipoleFieldGradient[particleJ.particleIndex][5] -= Eyz;

            OpenMM::Vec3 &dipolesJ = (*field.inducedDipoles)[particleJ.particleIndex];
            xDipole = dipolesJ[0];
            yDipole = dipolesJ[1];
            zDipole = dipolesJ[2];
            muDotR = xDipole*dx + yDipole*dy + zDipole*dz;
            Exx = muDotR*dx*dx*preFactor3 - (2.0*xDipole*dx + muDotR)*preFactor2;
            Eyy = muDotR*dy*dy*preFactor3 - (2.0*yDipole*dy + muDotR)*preFactor2;
            Ezz = muDotR*dz*dz*preFactor3 - (2.0*zDipole*dz + muDotR)*preFactor2;
            Exy = muDotR*dx*dy*preFactor3 - (xDipole*dy + yDipole*dx)*preFactor2;
            Exz = muDotR*dx*dz*preFactor3 - (xDipole*dz + zDipole*dx)*preFactor2;
            Eyz = muDotR*dy*dz*preFactor3 - (yDipole*dz + zDipole*dy)*preFactor2;

            field.inducedDipoleFieldGradient[particleI.particleIndex][0] += Exx;
            field.inducedDipoleFieldGradient[particleI.particleIndex][1] += Eyy;
            field.inducedDipoleFieldGradient[particleI.particleIndex][2] += Ezz;
            field.inducedDipoleFieldGradient[particleI.particleIndex][3] += Exy;
            field.inducedDipoleFieldGradient[particleI.particleIndex][4] += Exz;
            field.inducedDipoleFieldGradient[particleI.particleIndex][5] += Eyz;
        }
    }
}

double pGM_ReferencePmeMultipoleForce::calculatePmeSelfEnergy(const vector<MultipoleParticleData>& particleData) const
{
    double cii = 0.0;
    double dii = 0.0;
    double qii = 0.0;
    for (unsigned int ii = 0; ii < _numParticles; ii++) {

        const MultipoleParticleData& particleI = particleData[ii];

        cii += particleI.charge*particleI.charge;

        Vec3 dipole(particleI.sphericalDipole[1], particleI.sphericalDipole[2], particleI.sphericalDipole[0]);
        dii += dipole.dot(dipole + (_inducedDipole[ii]+_inducedDipolePolar[ii])*0.5);

    }
    double prefac = -_alphaEwald * _electric / (_dielectric*SQRT_PI);
    double a2 = _alphaEwald * _alphaEwald;
    double a4 = a2*a2;
    double energy = prefac*(cii + twoThirds*a2*dii + fourOverFifteen*a4*qii);
    return energy;
}


double pGM_ReferencePmeMultipoleForce::calculatePmeDirectElectrostaticPairIxn(const MultipoleParticleData& particleI,
                                                                                    const MultipoleParticleData& particleJ,
                                                                                    const vector<double>& scalingFactors,
                                                                                    vector<Vec3>& forces,
                                                                                    vector<Vec3>& torques) const
{

    unsigned int iIndex = particleI.particleIndex;
    unsigned int jIndex = particleJ.particleIndex;

    double energy;
    Vec3 deltaR = particleJ.position - particleI.position;
    getPeriodicDelta(deltaR);
    double r2 = deltaR.dot(deltaR);

    if (r2 > _cutoffDistanceSquared)
        return 0.0;

    double r = sqrt(r2);

    // Start by constructing rotation matrices to put dipoles and
    // quadrupoles into the QI frame, from the lab frame.
    double qiRotationMatrix1[3][3];
    formQIRotationMatrix(particleI.position, particleJ.position, deltaR, r, qiRotationMatrix1);
    double qiRotationMatrix2[5][5];
    buildSphericalQuadrupoleRotationMatrix(qiRotationMatrix1, qiRotationMatrix2);
    // The force rotation matrix rotates the QI forces into the lab
    // frame, and makes sure the result is in {x,y,z} ordering. Its
    // transpose is used to rotate the induced dipoles to the QI frame.
    double forceRotationMatrix[3][3];
    forceRotationMatrix[0][0] = qiRotationMatrix1[1][1];
    forceRotationMatrix[0][1] = qiRotationMatrix1[2][1];
    forceRotationMatrix[0][2] = qiRotationMatrix1[0][1];
    forceRotationMatrix[1][0] = qiRotationMatrix1[1][2];
    forceRotationMatrix[1][1] = qiRotationMatrix1[2][2];
    forceRotationMatrix[1][2] = qiRotationMatrix1[0][2];
    forceRotationMatrix[2][0] = qiRotationMatrix1[1][0];
    forceRotationMatrix[2][1] = qiRotationMatrix1[2][0];
    forceRotationMatrix[2][2] = qiRotationMatrix1[0][0];
    // For efficiency, we go ahead and cache that transposed version
    // now, because we need to do 4 rotations in total (I,J, and p,d).
    // We also fold in the factor of 0.5 needed to average the p and d
    // components.
    double inducedDipoleRotationMatrix[3][3];
    inducedDipoleRotationMatrix[0][0] = 0.5*qiRotationMatrix1[0][1];
    inducedDipoleRotationMatrix[0][1] = 0.5*qiRotationMatrix1[0][2];
    inducedDipoleRotationMatrix[0][2] = 0.5*qiRotationMatrix1[0][0];
    inducedDipoleRotationMatrix[1][0] = 0.5*qiRotationMatrix1[1][1];
    inducedDipoleRotationMatrix[1][1] = 0.5*qiRotationMatrix1[1][2];
    inducedDipoleRotationMatrix[1][2] = 0.5*qiRotationMatrix1[1][0];
    inducedDipoleRotationMatrix[2][0] = 0.5*qiRotationMatrix1[2][1];
    inducedDipoleRotationMatrix[2][1] = 0.5*qiRotationMatrix1[2][2];
    inducedDipoleRotationMatrix[2][2] = 0.5*qiRotationMatrix1[2][0];

    // Rotate the induced dipoles to the QI frame.
    double qiUindI[3], qiUindJ[3], qiUinpI[3], qiUinpJ[3];
    for (int ii = 0; ii < 3; ii++) {
        double valIP = 0.0;
        double valID = 0.0;
        double valJP = 0.0;
        double valJD = 0.0;
        for (int jj = 0; jj < 3; jj++) {
            valIP += inducedDipoleRotationMatrix[ii][jj] * _inducedDipolePolar[iIndex][jj];
            valID += inducedDipoleRotationMatrix[ii][jj] * _inducedDipole[iIndex][jj];
            valJP += inducedDipoleRotationMatrix[ii][jj] * _inducedDipolePolar[jIndex][jj];
            valJD += inducedDipoleRotationMatrix[ii][jj] * _inducedDipole[jIndex][jj];
        }
        qiUindI[ii] = valID;
        qiUinpI[ii] = valIP;
        qiUindJ[ii] = valJD;
        qiUinpJ[ii] = valJP;
    }

    // The Qtilde intermediates (QI frame multipoles) for atoms I and J
    double qiQI[9], qiQJ[9];
    // Rotate the permanent multipoles to the QI frame.
    qiQI[0] = particleI.charge;
    qiQJ[0] = particleJ.charge;
    for (int ii = 0; ii < 3; ii++) {
        double valI = 0.0;
        double valJ = 0.0;
        for (int jj = 0; jj < 3; jj++) {
            valI += qiRotationMatrix1[ii][jj] * particleI.sphericalDipole[jj];
            valJ += qiRotationMatrix1[ii][jj] * particleJ.sphericalDipole[jj];
        }
        qiQI[ii+1] = valI;
        qiQJ[ii+1] = valJ;
    }
    for (int ii = 0; ii < 5; ii++) {
        double valI = 0.0;
        double valJ = 0.0;
        for (int jj = 0; jj < 5; jj++) {
            valI += qiRotationMatrix2[ii][jj] * particleI.sphericalQuadrupole[jj];
            valJ += qiRotationMatrix2[ii][jj] * particleJ.sphericalQuadrupole[jj];
        }
        qiQI[ii+4] = valI;
        qiQJ[ii+4] = valJ;
    }

    // The Qtilde{x,y,z} torque intermediates for atoms I and J, which are used to obtain the torques on the permanent moments.
    double qiQIX[9] = {0.0, qiQI[3], 0.0, -qiQI[1], sqrtThree*qiQI[6], qiQI[8], -sqrtThree*qiQI[4] - qiQI[7], qiQI[6], -qiQI[5]};
    double qiQIY[9] = {0.0, -qiQI[2], qiQI[1], 0.0, -sqrtThree*qiQI[5], sqrtThree*qiQI[4] - qiQI[7], -qiQI[8], qiQI[5], qiQI[6]};
    double qiQIZ[9] = {0.0, 0.0, -qiQI[3], qiQI[2], 0.0, -qiQI[6], qiQI[5], -2.0*qiQI[8], 2.0*qiQI[7]};
    double qiQJX[9] = {0.0, qiQJ[3], 0.0, -qiQJ[1], sqrtThree*qiQJ[6], qiQJ[8], -sqrtThree*qiQJ[4] - qiQJ[7], qiQJ[6], -qiQJ[5]};
    double qiQJY[9] = {0.0, -qiQJ[2], qiQJ[1], 0.0, -sqrtThree*qiQJ[5], sqrtThree*qiQJ[4] - qiQJ[7], -qiQJ[8], qiQJ[5], qiQJ[6]};
    double qiQJZ[9] = {0.0, 0.0, -qiQJ[3], qiQJ[2], 0.0, -qiQJ[6], qiQJ[5], -2.0*qiQJ[8], 2.0*qiQJ[7]};

    // The field derivatives at I due to permanent and induced moments on J, and vice-versa.
    // Also, their derivatives w.r.t. R, which are needed for force calculations
    double Vij[9], Vji[9], VjiR[9], VijR[9];
    // The field derivatives at I due to only permanent moments on J, and vice-versa.
    double Vijp[3], Vijd[3], Vjip[3], Vjid[3];
    double rInvVec[7], alphaRVec[8], bVec[5];

    double prefac = (_electric/_dielectric);
    double rInv = 1.0 / r;

    // The rInvVec array is defined such that the ith element is R^-i, with the
    // dieleectric constant folded in, to avoid conversions later.
    rInvVec[1] = prefac * rInv;
    for (int i = 2; i < 7; ++i)
        rInvVec[i] = rInvVec[i-1] * rInv;

    // The alpharVec array is defined such that the ith element is (alpha R)^i,
    // where kappa (alpha in OpenMM parlance) is the Ewald attenuation parameter.
    alphaRVec[1] = _alphaEwald * r;
    for (int i = 2; i < 8; ++i)
        alphaRVec[i] = alphaRVec[i-1] * alphaRVec[1];

    double erfAlphaR = erf(alphaRVec[1]);
    double X = 2.0*exp(-alphaRVec[2])/SQRT_PI;
    double mScale = scalingFactors[M_SCALE];
    double dScale = scalingFactors[D_SCALE];
    double pScale = scalingFactors[P_SCALE];
    double uScale = scalingFactors[U_SCALE];

    int doubleFactorial = 1, facCount = 1;
    double tmp = alphaRVec[1];
    bVec[1] = -erfAlphaR;
    for (int i=2; i < 5; ++i) {
        bVec[i] = bVec[i-1] + tmp * X / doubleFactorial;
        facCount = facCount + 2;
        doubleFactorial = doubleFactorial * facCount;
        tmp *= 2.0 * alphaRVec[2];
    }

    double dmp = particleI.dampingFactor*particleJ.dampingFactor;
    double a = particleI.thole < particleJ.thole ? particleI.thole : particleJ.thole;
    double u = r/dmp;
    double au3 = fabs(dmp) > 1.0e-5f ? a*u*u*u : 0.0;
    double expau3 = fabs(dmp) > 1.0e-5f ? exp(-au3) : 0.0;
    double a2u6 = au3*au3;
    double a3u9 = a2u6*au3;
    // Thole damping factors for energies
    double thole_c  = 1.0 - expau3;
    double thole_d0 = 1.0 - expau3*(1.0 + 1.5*au3);
    double thole_d1 = 1.0 - expau3;
    double thole_q0 = 1.0 - expau3*(1.0 + au3 + a2u6);
    double thole_q1 = 1.0 - expau3*(1.0 + au3);
    // Thole damping factors for derivatives
    double dthole_c  = 1.0 - expau3*(1.0 + 1.5*au3);
    double dthole_d0 = 1.0 - expau3*(1.0 + au3 + 1.5*a2u6);
    double dthole_d1 = 1.0 - expau3*(1.0 + au3);
    double dthole_q0 = 1.0 - expau3*(1.0 + au3 + 0.25*a2u6 + 0.75*a3u9);
    double dthole_q1 = 1.0 - expau3*(1.0 + au3 + 0.75*a2u6);

    // Now we compute the (attenuated) Coulomb operator and its derivatives, contracted with
    // permanent moments and induced dipoles.  Note that the coefficient of the permanent force
    // terms is half of the expected value; this is because we compute the interaction of I with
    // the sum of induced and permanent moments on J, as well as the interaction of J with I's
    // permanent and induced moments; doing so double counts the permanent-permanent interaction.
    double ePermCoef, dPermCoef, eUIndCoef, dUIndCoef, eUInpCoef, dUInpCoef;

    // C-C terms (m=0)
    ePermCoef = rInvVec[1]*(mScale + bVec[2] - alphaRVec[1]*X);
    dPermCoef = -0.5*(mScale + bVec[2])*rInvVec[2];
    Vij[0]  = ePermCoef*qiQJ[0];
    Vji[0]  = ePermCoef*qiQI[0];
    VijR[0] = dPermCoef*qiQJ[0];
    VjiR[0] = dPermCoef*qiQI[0];

    // C-D and C-Uind terms (m=0)
    ePermCoef = rInvVec[2]*(mScale + bVec[2]);
    eUIndCoef = rInvVec[2]*(pScale*thole_c + bVec[2]);
    eUInpCoef = rInvVec[2]*(dScale*thole_c + bVec[2]);
    dPermCoef = -rInvVec[3]*(mScale + bVec[2] + alphaRVec[3]*X);
    dUIndCoef = -2.0*rInvVec[3]*(pScale*dthole_c + bVec[2] + alphaRVec[3]*X);
    dUInpCoef = -2.0*rInvVec[3]*(dScale*dthole_c + bVec[2] + alphaRVec[3]*X);
    Vij[0]  += -(ePermCoef*qiQJ[1] + eUIndCoef*qiUindJ[0] + eUInpCoef*qiUinpJ[0]);
    Vji[1]   = -(ePermCoef*qiQI[0]);
    VijR[0] += -(dPermCoef*qiQJ[1] + dUIndCoef*qiUindJ[0] + dUInpCoef*qiUinpJ[0]);
    VjiR[1]  = -(dPermCoef*qiQI[0]);
    Vjip[0]  = -(eUInpCoef*qiQI[0]);
    Vjid[0]  = -(eUIndCoef*qiQI[0]);
    // D-C and Uind-C terms (m=0)
    Vij[1]   = ePermCoef*qiQJ[0];
    Vji[0]  += ePermCoef*qiQI[1] + eUIndCoef*qiUindI[0] + eUInpCoef*qiUinpI[0];
    VijR[1]  = dPermCoef*qiQJ[0];
    VjiR[0] += dPermCoef*qiQI[1] + dUIndCoef*qiUindI[0] + dUInpCoef*qiUinpI[0];
    Vijp[0]  = eUInpCoef*qiQJ[0];
    Vijd[0]  = eUIndCoef*qiQJ[0];

    // D-D and D-Uind terms (m=0)
    ePermCoef = -twoThirds*rInvVec[3]*(3.0*(mScale + bVec[3]) + alphaRVec[3]*X);
    eUIndCoef = -twoThirds*rInvVec[3]*(3.0*(pScale*thole_d0 + bVec[3]) + alphaRVec[3]*X);
    eUInpCoef = -twoThirds*rInvVec[3]*(3.0*(dScale*thole_d0 + bVec[3]) + alphaRVec[3]*X);
    dPermCoef = rInvVec[4]*(3.0*(mScale + bVec[3]) + 2.*alphaRVec[5]*X);
    dUIndCoef = rInvVec[4]*(6.0*(pScale*dthole_d0 + bVec[3]) + 4.0*alphaRVec[5]*X);
    dUInpCoef = rInvVec[4]*(6.0*(dScale*dthole_d0 + bVec[3]) + 4.0*alphaRVec[5]*X);
    Vij[1]  += ePermCoef*qiQJ[1] + eUIndCoef*qiUindJ[0] + eUInpCoef*qiUinpJ[0];
    Vji[1]  += ePermCoef*qiQI[1] + eUIndCoef*qiUindI[0] + eUInpCoef*qiUinpI[0];
    VijR[1] += dPermCoef*qiQJ[1] + dUIndCoef*qiUindJ[0] + dUInpCoef*qiUinpJ[0];
    VjiR[1] += dPermCoef*qiQI[1] + dUIndCoef*qiUindI[0] + dUInpCoef*qiUinpI[0];
    Vijp[0] += eUInpCoef*qiQJ[1];
    Vijd[0] += eUIndCoef*qiQJ[1];
    Vjip[0] += eUInpCoef*qiQI[1];
    Vjid[0] += eUIndCoef*qiQI[1];
    // D-D and D-Uind terms (m=1)
    ePermCoef = rInvVec[3]*(mScale + bVec[3] - twoThirds*alphaRVec[3]*X);
    eUIndCoef = rInvVec[3]*(pScale*thole_d1 + bVec[3] - twoThirds*alphaRVec[3]*X);
    eUInpCoef = rInvVec[3]*(dScale*thole_d1 + bVec[3] - twoThirds*alphaRVec[3]*X);
    dPermCoef = -1.5*rInvVec[4]*(mScale + bVec[3]);
    dUIndCoef = -3.0*rInvVec[4]*(pScale*dthole_d1 + bVec[3]);
    dUInpCoef = -3.0*rInvVec[4]*(dScale*dthole_d1 + bVec[3]);
    Vij[2]  = ePermCoef*qiQJ[2] + eUIndCoef*qiUindJ[1] + eUInpCoef*qiUinpJ[1];
    Vji[2]  = ePermCoef*qiQI[2] + eUIndCoef*qiUindI[1] + eUInpCoef*qiUinpI[1];
    VijR[2] = dPermCoef*qiQJ[2] + dUIndCoef*qiUindJ[1] + dUInpCoef*qiUinpJ[1];
    VjiR[2] = dPermCoef*qiQI[2] + dUIndCoef*qiUindI[1] + dUInpCoef*qiUinpI[1];
    Vij[3]  = ePermCoef*qiQJ[3] + eUIndCoef*qiUindJ[2] + eUInpCoef*qiUinpJ[2];
    Vji[3]  = ePermCoef*qiQI[3] + eUIndCoef*qiUindI[2] + eUInpCoef*qiUinpI[2];
    VijR[3] = dPermCoef*qiQJ[3] + dUIndCoef*qiUindJ[2] + dUInpCoef*qiUinpJ[2];
    VjiR[3] = dPermCoef*qiQI[3] + dUIndCoef*qiUindI[2] + dUInpCoef*qiUinpI[2];
    Vijp[1] = eUInpCoef*qiQJ[2];
    Vijd[1] = eUIndCoef*qiQJ[2];
    Vjip[1] = eUInpCoef*qiQI[2];
    Vjid[1] = eUIndCoef*qiQI[2];
    Vijp[2] = eUInpCoef*qiQJ[3];
    Vijd[2] = eUIndCoef*qiQJ[3];
    Vjip[2] = eUInpCoef*qiQI[3];
    Vjid[2] = eUIndCoef*qiQI[3];



    // Evaluate the energies, forces and torques due to permanent+induced moments
    // interacting with just the permanent moments.
    energy = 0.5*(qiQI[0]*Vij[0] + qiQJ[0]*Vji[0]);
    double fIZ = qiQI[0]*VijR[0];
    double fJZ = qiQJ[0]*VjiR[0];
    double EIX = 0.0, EIY = 0.0, EIZ = 0.0, EJX = 0.0, EJY = 0.0, EJZ = 0.0;
    for (int i = 1; i < 9; ++i) {
        energy += 0.5*(qiQI[i]*Vij[i] + qiQJ[i]*Vji[i]);
        fIZ += qiQI[i]*VijR[i];
        fJZ += qiQJ[i]*VjiR[i];
        EIX += qiQIX[i]*Vij[i];
        EIY += qiQIY[i]*Vij[i];
        EIZ += qiQIZ[i]*Vij[i];
        EJX += qiQJX[i]*Vji[i];
        EJY += qiQJY[i]*Vji[i];
        EJZ += qiQJZ[i]*Vji[i];
    }

    // Define the torque intermediates for the induced dipoles. These are simply the induced dipole torque
    // intermediates dotted with the field due to permanent moments only, at each center. We inline the
    // induced dipole torque intermediates here, for simplicity. N.B. There are no torques on the dipoles
    // themselves, so we accumulate the torque intermediates into separate variables to allow them to be
    // used only in the force calculation.
    //
    // The torque about the x axis (needed to obtain the y force on the induced dipoles, below)
    //    qiUindIx[0] = qiQUindI[2];    qiUindIx[1] = 0;    qiUindIx[2] = -qiQUindI[0]
    double iEIX = qiUinpI[2]*Vijp[0] + qiUindI[2]*Vijd[0] - qiUinpI[0]*Vijp[2] - qiUindI[0]*Vijd[2];
    double iEJX = qiUinpJ[2]*Vjip[0] + qiUindJ[2]*Vjid[0] - qiUinpJ[0]*Vjip[2] - qiUindJ[0]*Vjid[2];
    // The torque about the y axis (needed to obtain the x force on the induced dipoles, below)
    //    qiUindIy[0] = -qiQUindI[1];   qiUindIy[1] = qiQUindI[0];    qiUindIy[2] = 0
    double iEIY = qiUinpI[0]*Vijp[1] + qiUindI[0]*Vijd[1] - qiUinpI[1]*Vijp[0] - qiUindI[1]*Vijd[0];
    double iEJY = qiUinpJ[0]*Vjip[1] + qiUindJ[0]*Vjid[1] - qiUinpJ[1]*Vjip[0] - qiUindJ[1]*Vjid[0];

    // Add in the induced-induced terms, if needed.
    if(getPolarizationType() == pGM_ReferenceMultipoleForce::Mutual) {
        // Uind-Uind terms (m=0)
        double eCoef = -fourThirds*rInvVec[3]*(3.0*(uScale*thole_d0 + bVec[3]) + alphaRVec[3]*X);
        double dCoef = rInvVec[4]*(6.0*(uScale*dthole_d0 + bVec[3]) + 4.0*alphaRVec[5]*X);
        iEIX += eCoef*(qiUinpI[2]*qiUindJ[0] + qiUindI[2]*qiUinpJ[0]);
        iEJX += eCoef*(qiUinpJ[2]*qiUindI[0] + qiUindJ[2]*qiUinpI[0]);
        iEIY -= eCoef*(qiUinpI[1]*qiUindJ[0] + qiUindI[1]*qiUinpJ[0]);
        iEJY -= eCoef*(qiUinpJ[1]*qiUindI[0] + qiUindJ[1]*qiUinpI[0]);
        fIZ += dCoef*(qiUinpI[0]*qiUindJ[0] + qiUindI[0]*qiUinpJ[0]);
        fJZ += dCoef*(qiUinpJ[0]*qiUindI[0] + qiUindJ[0]*qiUinpI[0]);
        // Uind-Uind terms (m=1)
        eCoef = 2.0*rInvVec[3]*(uScale*thole_d1 + bVec[3] - twoThirds*alphaRVec[3]*X);
        dCoef = -3.0*rInvVec[4]*(uScale*dthole_d1 + bVec[3]);
        iEIX -= eCoef*(qiUinpI[0]*qiUindJ[2] + qiUindI[0]*qiUinpJ[2]);
        iEJX -= eCoef*(qiUinpJ[0]*qiUindI[2] + qiUindJ[0]*qiUinpI[2]);
        iEIY += eCoef*(qiUinpI[0]*qiUindJ[1] + qiUindI[0]*qiUinpJ[1]);
        iEJY += eCoef*(qiUinpJ[0]*qiUindI[1] + qiUindJ[0]*qiUinpI[1]);
        fIZ += dCoef*(qiUinpI[1]*qiUindJ[1] + qiUindI[1]*qiUinpJ[1] + qiUinpI[2]*qiUindJ[2] + qiUindI[2]*qiUinpJ[2]);
        fJZ += dCoef*(qiUinpJ[1]*qiUindI[1] + qiUindJ[1]*qiUinpI[1] + qiUinpJ[2]*qiUindI[2] + qiUindJ[2]*qiUinpI[2]);
    }

    // The quasi-internal frame forces and torques.  Note that the induced torque intermediates are
    // used in the force expression, but not in the torques; the induced dipoles are isotropic.
    double qiForce[3] = {rInv*(EIY+EJY+iEIY+iEJY), -rInv*(EIX+EJX+iEIX+iEJX), -(fJZ+fIZ)};
    double qiTorqueI[3] = {-EIX, -EIY, -EIZ};
    double qiTorqueJ[3] = {-EJX, -EJY, -EJZ};

    // Rotate the forces and torques back to the lab frame
    for (int ii = 0; ii < 3; ii++) {
        double forceVal = 0.0;
        double torqueIVal = 0.0;
        double torqueJVal = 0.0;
        for (int jj = 0; jj < 3; jj++) {
            forceVal   += forceRotationMatrix[ii][jj] * qiForce[jj];
            torqueIVal += forceRotationMatrix[ii][jj] * qiTorqueI[jj];
            torqueJVal += forceRotationMatrix[ii][jj] * qiTorqueJ[jj];
        }
        torques[iIndex][ii] += torqueIVal;
        torques[jIndex][ii] += torqueJVal;
        forces[iIndex][ii]  -= forceVal;
        forces[jIndex][ii]  += forceVal;
    }
    return energy;

}

double pGM_ReferencePmeMultipoleForce::calculateElectrostatic(const vector<MultipoleParticleData>& particleData,
                                                                vector<Vec3>& torques, vector<Vec3>& forces)
{
    double energy = 0.0;
    vector<double> scaleFactors(LAST_SCALE_TYPE_INDEX);
    for (auto& s : scaleFactors)
        s = 1.0;

    // loop over particle pairs for direct space interactions

    for (unsigned int ii = 0; ii < particleData.size(); ii++) {
        for (unsigned int jj = ii+1; jj < particleData.size(); jj++) {

            if (jj <= _maxScaleIndex[ii]) {
                getMultipoleScaleFactors(ii, jj, scaleFactors);
            }

            energy += calculatePmeDirectElectrostaticPairIxn(particleData[ii], particleData[jj], scaleFactors, forces, torques);

            if (jj <= _maxScaleIndex[ii]) {
                for (auto& s : scaleFactors)
                    s = 1.0;
            }
        }
    }

    // The polarization energy
    calculatePmeSelfTorque(particleData, torques);
    energy += computeReciprocalSpaceInducedDipoleForceAndEnergy(getPolarizationType(), particleData, forces, torques);
    energy += computeReciprocalSpaceFixedMultipoleForceAndEnergy(particleData, forces, torques);
    energy += calculatePmeSelfEnergy(particleData);


    return energy;
}
