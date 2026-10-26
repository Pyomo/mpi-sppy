#include "xpress_params.hpp"

XpressParams::XpressParams(const std::string& jsonFileName)
  : SolverParams(jsonFileName)
{
  extrasWarn(canHave, "XpressParams constructor will not process extras: ");
}

void XpressParams::pushSolverParams(SolverInterface& scenario) {
  XpressInterface* s = dynamic_cast<XpressInterface*>(&scenario);
  if (!s) {
    return;  // Not an Xpress solver instance
  }

  if (auto v = getMipGap()) {
    s->setXpressDblControl(XPRS_MIPRELSTOP, *v);
  }
  if (auto v = getTimeLimit()) {
    s->setXpressDblControl(XPRS_TIMELIMIT, *v);
  }
  if (auto v = getThreads()) {
    s->setXpressIntControl(XPRS_THREADS, *v);
  }
}
