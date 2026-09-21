#!/usr/bin/env Rscript
# Step 4 -- join narratives to coded CRIS fields (§4.7).
#
# Reads the 18 columns §6 names from the 8 GB / 11.25 M-row person-level fst IN CHUNKS,
# derives the §4.7 indicator columns per person/unit row, aggregates to crash level inside
# each chunk, then folds the chunk aggregates together with any(). `any()` is associative,
# so a crash whose rows straddle a chunk boundary still aggregates correctly.
#
# Chunking is not an optimisation, it is required: a whole-file read needs several GB and
# this machine runs the Jev screen concurrently (a plain read_fst died with
# "cannot allocate vector of size 85.8 Mb").
#
# It aggregates ALL crashes rather than only the Stage-2 subset, so the result is computed
# once and reused by Stage 2, by the gold set, and by Paper 2. Output is small
# (~5.6 M rows x 17 logicals).
#
# Usage:
#   Rscript s04_join_fst.R --probe            # value distributions from one chunk
#   Rscript s04_join_fst.R [--chunk 400000]   # full aggregate -> coded_join.csv/.fst

suppressMessages({library(fst); library(data.table)})

# The crash-level fst lives outside the repository. Set JEV_CRIS_FST to point at it; the
# default is the path used for the run reported in the manuscript. DATA resolves from the
# script location so the repository can sit on any drive.
FST  <- Sys.getenv("JEV_CRIS_FST",
  "D:/OneDrive - Texas State University/Das, Subasish's files - 2027_TRBAM/CRIS/CrUnPPr_2017_2025_LimWithNarr2_95VarV01.fst")
SRC  <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
DATA <- file.path(dirname(SRC), "data")
args  <- commandArgs(trailingOnly = TRUE)
PROBE <- "--probe" %in% args
CHUNK <- if ("--chunk" %in% args) as.integer(args[which(args == "--chunk") + 1]) else 400000L

COLS <- c("Crash_ID","Contrib_Factr_1_ID","Contrib_Factr_2_ID","Prsn_Alc_Rslt_ID",
          "Prsn_Bac_Test_Rslt","Prsn_Drg_Rslt_ID","Prsn_Rest_ID","Harm_Evnt_ID",
          "Surf_Cond_ID","Wthr_Cond_ID","Othr_Factr_ID","FHE_Collsn_ID","Crash_Sev_ID",
          "Prsn_Injry_Sev_ID","Tot_Injry_Cnt","Death_Cnt","Unit_Desc_ID","Prsn_Type_ID")

meta <- fst::metadata_fst(FST)
NROW <- meta$nrOfRows
cat(sprintf("fst rows: %s  chunk: %s\n", format(NROW, big.mark=","), format(CHUNK, big.mark=",")))

derive <- function(d) {
  for (cc in c("Contrib_Factr_1_ID","Contrib_Factr_2_ID","Othr_Factr_ID","Prsn_Rest_ID",
               "Harm_Evnt_ID","Surf_Cond_ID","Wthr_Cond_ID","FHE_Collsn_ID","Crash_Sev_ID",
               "Prsn_Injry_Sev_ID"))
    set(d, j = cc, value = trimws(as.character(d[[cc]])))
  cf <- function(p) grepl(p, d$Contrib_Factr_1_ID, ignore.case=TRUE) |
                    grepl(p, d$Contrib_Factr_2_ID, ignore.case=TRUE)
  d[, `:=`(
    cf_alcohol  = cf("Under Influence - Alcohol|Had Been Drinking"),
    cf_drug     = cf("Under Influence - Drug"),
    cf_fatigue  = cf("Fatigued Or Asleep"),
    cf_ill      = cf("Ill [(]Explain"),           # driver medical episode, coded
    cf_animal   = cf("Animal On Road"),
    cf_phone    = cf("Cell/Mobile Device Use"),
    alc_rslt    = !is.na(Prsn_Alc_Rslt_ID) & Prsn_Alc_Rslt_ID == 1,
    bac_pos     = !is.na(Prsn_Bac_Test_Rslt) & Prsn_Bac_Test_Rslt > 0,
    drg_rslt    = !is.na(Prsn_Drg_Rslt_ID) & Prsn_Drg_Rslt_ID == 1,
    rest_none   = trimws(Prsn_Rest_ID) == "None",   # verified: the only unbelted code; "Not Applicable" must not match
    harm_animal = grepl("Animal", Harm_Evnt_ID, ignore.case=TRUE),
    surf_wet    = grepl("Wet|Standing Water", Surf_Cond_ID, ignore.case=TRUE),
    surf_ice    = grepl("Ice|Snow|Sleet|Slush", Surf_Cond_ID, ignore.case=TRUE),
    surf_dry    = grepl("^Dry$", Surf_Cond_ID, ignore.case=TRUE),
    othr_skid   = grepl("Lost Control Or Skidded", Othr_Factr_ID, ignore.case=TRUE),
    wthr_rain   = grepl("Rain", Wthr_Cond_ID, ignore.case=TRUE),
    fhe_opposite= grepl("Opposite Direction", FHE_Collsn_ID, ignore.case=TRUE),
    cf_wrongway = cf("Wrong Way - One Way Road|Wrong Side - Not Passing|Wrong Side - Approach"),
    injured     = grepl("Incapacitating|Possible Injury|Non-Incapacitating", Prsn_Injry_Sev_ID, ignore.case=TRUE),
    killed      = grepl("Killed|Fatal", Prsn_Injry_Sev_ID, ignore.case=TRUE)
  )]
  d[, .(
    coded_alcohol        = any(cf_alcohol | alc_rslt | bac_pos, na.rm=TRUE),
    coded_drug           = any(cf_drug | drg_rslt, na.rm=TRUE),
    coded_fatigue        = any(cf_fatigue, na.rm=TRUE),
    coded_medical        = any(cf_ill, na.rm=TRUE),
    coded_animal         = any(cf_animal | harm_animal, na.rm=TRUE),
    coded_phone          = any(cf_phone, na.rm=TRUE),
    coded_unbelted       = any(rest_none, na.rm=TRUE),
    coded_surf_wet       = any(surf_wet, na.rm=TRUE),
    coded_surf_ice       = any(surf_ice, na.rm=TRUE),
    coded_surf_dry       = any(surf_dry, na.rm=TRUE),
    coded_skid           = any(othr_skid, na.rm=TRUE),
    coded_hydro_proxy    = any(othr_skid & (surf_wet | wthr_rain), na.rm=TRUE),
    coded_wrongway       = any(cf_wrongway, na.rm=TRUE),
    coded_wrongway_proxy = any(fhe_opposite, na.rm=TRUE),
    coded_injured        = any(injured, na.rm=TRUE),
    coded_killed         = any(killed, na.rm=TRUE),
    tot_injry            = suppressWarnings(max(Tot_Injry_Cnt, na.rm=TRUE)),
    death_cnt            = suppressWarnings(max(Death_Cnt, na.rm=TRUE)),
    n_person_rows        = .N
  ), by = Crash_ID]
}

if (PROBE) {
  d <- as.data.table(read_fst(FST, columns = COLS, from = 1, to = min(CHUNK, NROW)))
  show <- function(cc, n = 40) {
    v <- sort(table(trimws(as.character(d[[cc]]))), decreasing = TRUE)
    cat("\n====", cc, "(", length(v), "distinct )\n"); print(utils::head(v, n))
  }
  for (cc in c("Contrib_Factr_1_ID","Othr_Factr_ID","Prsn_Rest_ID","Harm_Evnt_ID",
               "Surf_Cond_ID","Prsn_Injry_Sev_ID","FHE_Collsn_ID")) show(cc)
  cat("\nPrsn_Alc_Rslt_ID:\n"); print(table(d$Prsn_Alc_Rslt_ID, useNA="ifany"))
  cat("\nPrsn_Drg_Rslt_ID:\n"); print(table(d$Prsn_Drg_Rslt_ID, useNA="ifany"))
  quit(save = "no")
}

parts <- list(); i <- 1L; from <- 1L; t0 <- Sys.time()
while (from <= NROW) {
  to <- min(from + CHUNK - 1L, NROW)
  d <- as.data.table(read_fst(FST, columns = COLS, from = from, to = to))
  parts[[i]] <- derive(d)
  rm(d); gc(verbose = FALSE)
  cat(sprintf("  chunk %2d  rows %s-%s  crashes %s  %.0fs\n", i,
              format(from, big.mark=","), format(to, big.mark=","),
              format(nrow(parts[[i]]), big.mark=","),
              as.numeric(difftime(Sys.time(), t0, units="secs"))))
  from <- to + 1L; i <- i + 1L
}

all <- rbindlist(parts); rm(parts); gc(verbose = FALSE)
bool_cols <- grep("^coded_", names(all), value = TRUE)
agg <- all[, c(lapply(.SD, any),
               list(tot_injry = max(tot_injry), death_cnt = max(death_cnt),
                    n_person_rows = sum(n_person_rows))),
           by = Crash_ID, .SDcols = bool_cols]
rm(all); gc(verbose = FALSE)

cat(sprintf("\naggregated to %s crashes\n", format(nrow(agg), big.mark=",")))
cat("\ncoded-field prevalence over ALL crashes:\n")
for (cc in bool_cols)
  cat(sprintf("  %-22s %7.4f%%  (n=%s)\n", cc, 100*mean(agg[[cc]]),
              format(sum(agg[[cc]]), big.mark=",")))

write_fst(agg, file.path(DATA, "coded_join.fst"), compress = 80)
fwrite(agg, file.path(DATA, "coded_join.csv"))
cat("\nwrote coded_join.fst and coded_join.csv\n")
