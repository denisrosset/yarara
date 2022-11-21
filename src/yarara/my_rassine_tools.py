from __future__ import annotations

import glob as glob
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy import float64, int64, ndarray
from pandas.core.frame import DataFrame

from . import io, sts
from .analysis import tableXY
from .paths import root

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


# =============================================================================
# PRODUCE THE DACE TABLE SUMMARIZING RV TIMESERIES
# =============================================================================


class spec_time_series(object):
    def __init__(self, directory: str) -> None:
        if len(directory.split("/")) == 1:
            directory = root + "/Yarara/" + directory + "/data/s1d/HARPS03/WORKSPACE/"
        if directory[-1] != "/":
            directory = directory + "/"

        self.directory = directory
        self.starname = directory.split("/Yarara/")[-1].split("/")[0].split("=")[0]
        self.dir_root = directory.split("WORKSPACE/")[0]
        self.dir_yarara = directory.split("Yarara/")[0] + "Yarara/"
        self.cmap = "plasma"
        self.low_cmap = -0.005
        self.high_cmap = 0.005
        self.zoom = 1
        self.smooth_map = 1
        self.planet = False
        self.sp_type = None
        self.rv_sys = None
        self.teff = None
        self.log_g = None
        self.bv = None
        self.fwhm = None
        self.wave = None
        self.infos = {}
        self.ram = []

        self.dico_actif = "matching_diff"

        self.all_dicos = [
            "matching_diff",
            "matching_cosmics",
            "matching_fourier",
            "matching_telluric",
            "matching_oxy_bands",
            "matching_oxygen",
            "matching_pca",
            "matching_activity",
            "matching_ghost_a",
            "matching_ghost_b",
            "matching_database",
            "matching_stitching",
            "matching_berv",
            "matching_thar",
            "matching_contam",
            "matching_smooth",
            "matching_profile",
            "matching_mad",
        ]

        self.light_dicos = [
            "matching_diff",
            "matching_pca",
            "matching_activity",
            "matching_mad",
        ]

        self.planet_fitted = {}

        if not os.path.exists(self.directory + "Analyse_ccf.p"):
            ccf_summary = {"star_info": {"name": self.starname}}
            io.pickle_dump(ccf_summary, open(self.directory + "Analyse_ccf.p", "wb"))

        if not os.path.exists(self.directory + "Analyse_material.p"):
            file = pd.read_pickle(glob.glob(self.directory + "RASSI*.p")[0])
            wave = file["wave"]
            dico = {
                "wave": wave,
                "correction_factor": np.ones(len(wave)),
                "reference_spectrum": np.ones(len(wave)),
                "color_template": np.ones(len(wave)),
                "blaze_correction": np.ones(len(wave)),
                "rejected": np.zeros(len(wave)),
            }
            dico = pd.DataFrame(dico)
            io.pickle_dump(dico, open(self.directory + "Analyse_material.p", "wb"))

        if os.path.exists(self.directory + "/RASSINE_Master_spectrum.p"):
            master = pd.read_pickle(self.directory + "/RASSINE_Master_spectrum.p")
            master = master["flux"] / master["output"]["continuum_linear"]
            os.system(
                "mv "
                + self.directory
                + "RASSINE_Master_spectrum.p "
                + self.directory
                + "ras_Master_spectrum.p "
            )

        if not os.path.exists(self.dir_root + "IMAGES/"):
            os.system("mkdir " + self.dir_root + "IMAGES/")

        if not os.path.exists(self.dir_root + "CCF_MASK/"):
            os.system("mkdir " + self.dir_root + "CCF_MASK/")

        if not os.path.exists(self.dir_root + "PCA/"):
            os.system("mkdir " + self.dir_root + "PCA/")

        if not os.path.exists(self.dir_root + "REDUCTION_INFO/"):
            os.system("mkdir " + self.dir_root + "REDUCTION_INFO/")

        if not os.path.exists(self.dir_root + "STAR_INFO/"):
            os.system("mkdir " + self.dir_root + "STAR_INFO/")

        if not os.path.exists(self.dir_root + "STAR_INFO/Stellar_info_" + self.starname + ".p"):
            dico = {
                "Name": self.starname,
                "Simbad_name": {"fixed": "-"},
                "Sp_type": {"fixed": "G2V"},
                "Ra": {"fixed": "00 00 00.0000"},
                "Dec": {"fixed": "00 00 00.0000"},
                "Pma": {"fixed": 0.0},
                "Pmd": {"fixed": 0.0},
                "Rv_sys": {"fixed": 0.0},
                "Mstar": {"fixed": 1.0},
                "Rstar": {"fixed": 1.0},
                "magU": {"fixed": -26.0},
                "magB": {"fixed": -26.2},
                "magV": {"fixed": -26.8},
                "magR": {"fixed": -26.8},
                "UB": {"fixed": 0.0},
                "BV": {"fixed": 0.6},
                "VR": {"fixed": 0.0},
                "Dist_pc": {"fixed": 0.0},
                "Teff": {"fixed": 5775},
                "Log_g": {"fixed": 4.5},
                "FeH": {"fixed": 0.0},
                "Vsini": {"fixed": 2.0},
                "Vmicro": {"fixed": 1.0},
                "Prot": {"fixed": 25},
                "Pmag": {"fixed": 11},
                "FWHM": {"fixed": 6.0},
                "Contrast": {"fixed": 0.5},
                "CCF_delta": {"fixed": 5},
                "stellar_template": {"fixed": "MARCS_T5750_g4.5"},
            }

            io.pickle_dump(
                dico,
                open(
                    self.dir_root + "STAR_INFO/Stellar_info_" + self.starname + ".p",
                    "wb",
                ),
            )

        self.import_star_info()
        sp = self.star_info["Sp_type"]["fixed"][0]
        self.mask_harps = ["G2", "K5", "M2"][int((sp == "K") | (sp == "M")) + int(sp == "M")]
        try:
            self.fwhm = self.star_info["FWHM"]["YARARA"]
        except:
            self.fwhm = self.star_info["FWHM"]["fixed"]

        try:
            self.contrast = self.star_info["Contrast"]["YARARA"]
        except:
            self.contrast = self.star_info["Contrast"]["fixed"]

        if not os.path.exists(self.dir_root + "KEPLERIAN/"):
            os.system("mkdir " + self.dir_root + "KEPLERIAN/")

        if not os.path.exists(self.dir_root + "KEPLERIAN/MCMC/"):
            os.system("mkdir " + self.dir_root + "KEPLERIAN/MCMC/")

        if not os.path.exists(self.dir_root + "PERIODOGRAM/"):
            os.system("mkdir " + self.dir_root + "PERIODOGRAM/")

        if not os.path.exists(self.dir_root + "FILM"):
            os.system("mkdir " + self.dir_root + "FILM")

        if not os.path.exists(self.dir_root + "DETECTION_LIMIT/"):
            os.system("mkdir " + self.dir_root + "DETECTION_LIMIT/")

        if not os.path.exists(self.dir_root + "CORRECTION_MAP/"):
            os.system("mkdir " + self.dir_root + "CORRECTION_MAP/")

    def import_rassine_output(
        self: spec_time_series, return_name: bool = False, kw1: None = None, kw2: None = None
    ) -> Any:
        return sts.io.import_rassine_output(self, return_name, kw1, kw2)

    def import_star_info(self: spec_time_series) -> None:
        return sts.io.import_star_info(self)

    def import_table(self: spec_time_series) -> None:
        return sts.io.import_table(self)

    def import_material(self: spec_time_series) -> None:
        return sts.io.import_material(self)

    def import_dico_tree(self: spec_time_series) -> None:
        return sts.io.import_dico_tree(self)

    def import_spectrum(self: spec_time_series, num: Optional[int64] = None) -> Dict[str, Any]:
        return sts.io.import_spectrum(self, num)

    def yarara_star_info(
        self: spec_time_series,
        Rv_sys: None = None,
        simbad_name: None = None,
        magB: None = None,
        magV: None = None,
        magR: None = None,
        BV: None = None,
        VR: None = None,
        sp_type: None = None,
        Mstar: None = None,
        Rstar: None = None,
        Vsini: None = None,
        Vmicro: None = None,
        Teff: None = None,
        log_g: None = None,
        FeH: None = None,
        Prot: None = None,
        Fwhm: Optional[List[Union[str, float64]]] = None,
        Contrast: Optional[List[Union[str, float64]]] = None,
        CCF_delta: None = None,
        Pmag: None = None,
        stellar_template: None = None,
    ) -> None:
        return sts.io.yarara_star_info(
            self,
            Rv_sys,
            simbad_name,
            magB,
            magV,
            magR,
            BV,
            VR,
            sp_type,
            Mstar,
            Rstar,
            Vsini,
            Vmicro,
            Teff,
            log_g,
            FeH,
            Prot,
            Fwhm,
            Contrast,
            CCF_delta,
            Pmag,
            stellar_template,
        )

    def yarara_master_ccf(
        self: spec_time_series,
        sub_dico: str = "matching_diff",
        name_ext: str = "",
        rvs: Optional[ndarray] = None,
    ) -> None:
        return sts.limbo.yarara_master_ccf(self, sub_dico, name_ext, rvs)

    def yarara_poissonian_noise(
        self: spec_time_series,
        noise_wanted: float = 1 / 100,
        wave_ref: None = None,
        flat_snr: bool = True,
        seed: int = 9,
    ) -> Tuple[ndarray, ndarray]:
        return sts.util.yarara_poissonian_noise(self, noise_wanted, wave_ref, flat_snr, seed)

    def yarara_obs_info(
        self: spec_time_series,
        kw: DataFrame = [None, None],
        jdb: None = None,
        berv: None = None,
        rv: None = None,
        airmass: None = None,
        texp: None = None,
        seeing: None = None,
        humidity: None = None,
    ) -> None:
        return sts.io.yarara_obs_info(self, kw, jdb, berv, rv, airmass, texp, seeing, humidity)

    def yarara_get_orders(self: spec_time_series) -> ndarray:
        return sts.extract.yarara_get_orders(self)

    def yarara_get_pixels(self: spec_time_series) -> ndarray:
        return sts.extract.yarara_get_pixels(self)

    def supress_time_spectra(
        self: spec_time_series,
        liste: Optional[ndarray] = None,
        jdb_min: None = None,
        jdb_max: None = None,
        num_min: None = None,
        num_max: None = None,
        supress: bool = False,
        name_ext: str = "temp",
    ) -> None:
        return sts.io.supress_time_spectra(
            self, liste, jdb_min, jdb_max, num_min, num_max, supress, name_ext
        )

    def yarara_analyse_summary(self: spec_time_series, rm_old: bool = False) -> None:
        return sts.io.yarara_analyse_summary(self, rm_old)

    def yarara_get_berv_value(
        self: spec_time_series,
        time_value: float,
        Draw: bool = False,
        new: bool = True,
        light_graphic: bool = False,
        save_fig: bool = True,
    ) -> float64:
        return sts.extract.yarara_get_berv_value(
            self, time_value, Draw, new, light_graphic, save_fig
        )

    def yarara_non_zero_flux(
        self: spec_time_series, spectrum: Optional[ndarray] = None, min_value: None = None
    ) -> ndarray:
        return sts.util.yarara_non_zero_flux(self, spectrum, min_value)

    def yarara_median_master_backup(
        self,
        sub_dico: Optional[str] = "matching_diff",
        method: str = "mean",
        continuum: str = "linear",
        supress_telluric: bool = True,
        shift_spectrum: bool = False,
        telluric_tresh: float = 0.001,
        wave_min: int = 5750,
        wave_max: int = 5900,
        jdb_range: List[int] = [-100000, 100000, 1],
        mask_percentile: List[Optional[int]] = [None, 50],
        save: bool = True,
    ) -> None:
        return sts.util.yarara_median_master_backup(
            self,
            sub_dico,
            method,
            continuum,
            supress_telluric,
            shift_spectrum,
            telluric_tresh,
            wave_min,
            wave_max,
            jdb_range,
            mask_percentile,
            save,
        )

    def yarara_median_master(
        self: spec_time_series,
        sub_dico: Optional[str] = "matching_diff",
        continuum: str = "linear",
        method: str = "max",
        smooth_box: int = 7,
        supress_telluric: bool = True,
        shift_spectrum: bool = False,
        wave_min: int = 5750,
        wave_max: int = 5900,
        bin_berv: int = 10,
        bin_snr: Optional[int] = None,
        telluric_tresh: float = 0.001,
        jdb_range: List[int] = [-100000, 100000, 1],
        mask_percentile: List[Optional[int]] = [None, 50],
        save: bool = True,
    ) -> None:
        return sts.util.yarara_median_master(
            self,
            sub_dico,
            continuum,
            method,
            smooth_box,
            supress_telluric,
            shift_spectrum,
            wave_min,
            wave_max,
            bin_berv,
            bin_snr,
            telluric_tresh,
            jdb_range,
            mask_percentile,
            save,
        )

    def yarara_cut_spectrum(
        self: spec_time_series, wave_min: None = None, wave_max: Optional[int] = None
    ) -> None:
        return sts.util.yarara_cut_spectrum(self, wave_min, wave_max)

    def yarara_activity_index(
        self: spec_time_series,
        sub_dico: str = "matching_diff",
        continuum: str = "linear",
        plot: bool = True,
        debug: bool = False,
        calib_std: int = 0,
        optimize: bool = False,
        substract_map: List[Any] = [],
        add_map: List[Any] = [],
        p_noise: float = 1 / np.inf,
        save: bool = True,
    ) -> None:
        return sts.processing.yarara_activity_index(
            self,
            sub_dico,
            continuum,
            plot,
            debug,
            calib_std,
            optimize,
            substract_map,
            add_map,
            p_noise,
            save,
        )

    def yarara_telluric(
        self: spec_time_series,
        sub_dico: str = "matching_anchors",
        continuum: str = "linear",
        suppress_broad: bool = True,
        delta_window: int = 5,
        mask: Optional[str] = None,
        weighted: bool = False,
        reference: str = True,
        display_ccf: bool = False,
        ratio: bool = False,
        normalisation: str = "slope",
        ccf_oversampling: int = 3,
        wave_max: None = None,
        wave_min: None = None,
    ) -> None:
        return sts.telluric.yarara_telluric(
            self,
            sub_dico,
            continuum,
            suppress_broad,
            delta_window,
            mask,
            weighted,
            reference,
            display_ccf,
            ratio,
            normalisation,
            ccf_oversampling,
            wave_max,
            wave_min,
        )

    def yarara_ccf(
        self: spec_time_series,
        sub_dico: str = "matching_diff",
        continuum: str = "linear",
        mask: Optional[Union[ndarray, str]] = None,
        mask_name: None = None,
        ccf_name: None = None,
        mask_col: str = "weight_rv",
        treshold_telluric: int = 1,
        ratio: bool = False,
        element: None = None,
        reference: Union[bool, str] = True,
        weighted: bool = True,
        plot: bool = False,
        display_ccf: bool = False,
        save: bool = True,
        save_ccf_profile: bool = False,
        normalisation: str = "left",
        del_outside_max: bool = False,
        bis_analysis: bool = False,
        ccf_oversampling: int = 1,
        rv_range: Optional[int] = None,
        rv_borders: Optional[int] = None,
        delta_window: int = 5,
        debug: bool = False,
        rv_sys: Optional[int] = None,
        rv_shift: Optional[ndarray] = None,
        speed_up: bool = True,
        force_brute: bool = False,
        wave_min: None = None,
        wave_max: None = None,
        squared: bool = True,
        p_noise: float = 1 / np.inf,
        substract_map: List[Any] = [],
        add_map: List[Any] = [],
    ) -> Dict[str, tableXY]:
        return sts.processing.yarara_ccf(
            self,
            sub_dico,
            continuum,
            mask,
            mask_name,
            ccf_name,
            mask_col,
            treshold_telluric,
            ratio,
            element,
            reference,
            weighted,
            plot,
            display_ccf,
            save,
            save_ccf_profile,
            normalisation,
            del_outside_max,
            bis_analysis,
            ccf_oversampling,
            rv_range,
            rv_borders,
            delta_window,
            debug,
            rv_sys,
            rv_shift,
            speed_up,
            force_brute,
            wave_min,
            wave_max,
            squared,
            p_noise,
            substract_map,
            add_map,
        )

    def yarara_map(
        self: spec_time_series,
        sub_dico: str = "matching_diff",
        continuum: str = "linear",
        planet: bool = False,
        modulo: None = None,
        unit: float = 1.0,
        wave_min: None = 4000,
        wave_max: None = 4300,
        time_min: None = None,
        time_max: None = None,
        index: str = "index",
        ratio: bool = False,
        reference: bool = "median",
        berv_shift: bool = False,
        rv_shift: bool = False,
        new: bool = True,
        Plot: bool = True,
        substract_map: List[Any] = [],
        add_map: List[Any] = [],
        correction_factor: bool = True,
        p_noise: float = 1 / np.inf,
    ) -> Tuple[ndarray, ndarray, ndarray]:
        return sts.processing.yarara_map(
            self,
            sub_dico,
            continuum,
            planet,
            modulo,
            unit,
            wave_min,
            wave_max,
            time_min,
            time_max,
            index,
            ratio,
            reference,
            berv_shift,
            rv_shift,
            new,
            Plot,
            substract_map,
            add_map,
            correction_factor,
            p_noise,
        )

    def yarara_correct_pattern(
        self: spec_time_series,
        sub_dico: str = "matching_diff",
        continuum: str = "linear",
        wave_min: int = 6000,
        wave_max: int = 6100,
        reference: str = "median",
        width_range: List[float] = [0.1, 20],
        correct_blue: bool = True,
        correct_red: bool = True,
        jdb_range: Optional[List[int]] = None,
    ) -> None:
        return sts.instrument.yarara_correct_pattern(
            self,
            sub_dico,
            continuum,
            wave_min,
            wave_max,
            reference,
            width_range,
            correct_blue,
            correct_red,
            jdb_range,
        )

    def yarara_correct_smooth(
        self: spec_time_series,
        sub_dico: str = "matching_diff",
        continuum: str = "linear",
        reference: str = "median",
        wave_min: int = 4200,
        wave_max: int = 4300,
        window_ang: int = 5,
    ) -> None:
        return sts.outliers.yarara_correct_smooth(
            self, sub_dico, continuum, reference, wave_min, wave_max, window_ang
        )

    def yarara_retropropagation_correction(
        self: spec_time_series,
        correction_map: str = "matching_smooth",
        sub_dico: str = "matching_cosmics",
        continuum: str = "linear",
    ) -> None:
        return sts.processing.yarara_retropropagation_correction(
            self, correction_map, sub_dico, continuum
        )

    def yarara_correct_telluric_proxy(
        self: spec_time_series,
        sub_dico: str = "matching_fourier",
        sub_dico_output: str = "telluric",
        continuum: str = "linear",
        wave_min: int = 5700,
        wave_max: int = 5900,
        reference: str = "master",
        berv_shift: str = "berv",
        smooth_corr: int = 1,
        proxies_corr: List[str] = ["h2o_depth", "h2o_fwhm"],
        proxies_detrending: None = None,
        wave_min_correction: int = 4400,
        wave_max_correction: None = None,
        min_r_corr: float = 0.40,
        sigma_ext: int = 2,
    ) -> None:
        return sts.telluric.yarara_correct_telluric_proxy(
            self,
            sub_dico,
            sub_dico_output,
            continuum,
            wave_min,
            wave_max,
            reference,
            berv_shift,
            smooth_corr,
            proxies_corr,
            proxies_detrending,
            wave_min_correction,
            wave_max_correction,
            min_r_corr,
            sigma_ext,
        )

    def yarara_correct_oxygen(
        self: spec_time_series,
        sub_dico: str = "matching_telluric",
        continuum: str = "linear",
        berv_shift: str = "berv",
        reference: str = "master",
        wave_min: int = 5760,
        wave_max: int = 5850,
        oxygene_bands: List[List[int]] = [[5787, 5835], [6275, 6340], [6800, 6950]],
    ) -> None:
        return sts.telluric.yarara_correct_oxygen(
            self, sub_dico, continuum, berv_shift, reference, wave_min, wave_max, oxygene_bands
        )

    def yarara_correct_telluric_gradient(
        self: spec_time_series,
        sub_dico_detection: str = "matching_fourier",
        sub_dico_correction: None = "matching_oxygen",
        continuum: str = "linear",
        wave_min_train: int = 4200,
        wave_max_train: int = 5000,
        wave_min_correction: int = 4400,
        wave_max_correction: int = 6600,
        smooth_map: int = 1,
        berv_shift: str = "berv",
        reference: str = "master",
        inst_resolution: int = 110000,
        debug: bool = False,
        equal_weight: bool = True,
        nb_pca_comp: int = 20,
        nb_pca_comp_kept: None = None,
        nb_pca_max_kept: int = 5,
        calib_std: float = 1e-3,
    ) -> None:
        return sts.telluric.yarara_correct_telluric_gradient(
            self,
            sub_dico_detection,
            sub_dico_correction,
            continuum,
            wave_min_train,
            wave_max_train,
            wave_min_correction,
            wave_max_correction,
            smooth_map,
            berv_shift,
            reference,
            inst_resolution,
            debug,
            equal_weight,
            nb_pca_comp,
            nb_pca_comp_kept,
            nb_pca_max_kept,
            calib_std,
        )

    def yarara_correct_activity(
        self: spec_time_series,
        sub_dico: str = "matching_telluric",
        continuum: str = "linear",
        wave_min: int = 3900,
        wave_max: int = 4400,
        smooth_corr: int = 5,
        reference: str = "median",
        rv_shift: str = "none",
        proxy_corr: List[str] = ["CaII"],
    ) -> None:
        return sts.activity.yarara_correct_activity(
            self,
            sub_dico,
            continuum,
            wave_min,
            wave_max,
            smooth_corr,
            reference,
            rv_shift,
            proxy_corr,
        )

    def yarara_correct_cosmics(
        self: spec_time_series,
        sub_dico: str = "matching_diff",
        continuum: str = "linear",
        k_sigma: int = 3,
        bypass_warning: bool = True,
    ) -> None:
        return sts.outliers.yarara_correct_cosmics(
            self, sub_dico, continuum, k_sigma, bypass_warning
        )

    def yarara_correct_mad(
        self: spec_time_series,
        sub_dico: str = "matching_diff",
        continuum: str = "linear",
        k_sigma: int = 2,
        k_mad: int = 2,
        n_iter: int = 1,
        ext: str = "0",
    ) -> None:
        return sts.outliers.yarara_correct_mad(
            self, sub_dico, continuum, k_sigma, k_mad, n_iter, ext
        )

    def yarara_produce_mask_contam(
        self: spec_time_series, frog_file: str = root + "/Python/Material/Contam_HARPN.p"
    ) -> None:
        return sts.instrument.yarara_produce_mask_contam(self, frog_file)

    def yarara_produce_mask_frog(
        self: spec_time_series, frog_file: str = root + "/Python/Material/Ghost_HARPS03.p"
    ) -> None:
        #Frog file = Dictionnary containing a 2D wave matrix and 2D contaminations images
        return sts.instrument.yarara_produce_mask_frog(self, frog_file)

    def yarara_correct_frog(
        self: spec_time_series,
        sub_dico: str = "matching_diff",
        continuum: str = "linear",
        correction: str = "stitching",
        berv_shift: str = False,
        wave_min: int = 3800,
        wave_max: int = 3975,
        wave_min_train: int = 3700,
        wave_max_train: int = 6000,
        complete_analysis: bool = False,
        reference: str = "median",
        equal_weight: bool = True,
        nb_pca_comp: int = 10,
        pca_comp_kept: Optional[int] = None,
        rcorr_min: int = 0,
        treshold_contam: Union[int, float] = 0.5,
        algo_pca: str = "empca",
    ) -> None:
        return sts.instrument.yarara_correct_frog(
            self,
            sub_dico,
            continuum,
            correction,
            berv_shift,
            wave_min,
            wave_max,
            wave_min_train,
            wave_max_train,
        )

    def yarara_correct_borders_pxl(
        self: spec_time_series,
        pixels_to_reject: ndarray = [2, 4095],
        min_shift: int = -30,
        max_shift: int = 30,
    ) -> None:
        return sts.instrument.yarara_correct_borders_pxl(
            self, pixels_to_reject, min_shift, max_shift
        )

    def yarara_correct_brute(
        self: spec_time_series,
        sub_dico: str = "matching_mad",
        continuum: str = "linear",
        reference: str = "median",
        win_roll: int = 1000,
        min_length: int = 5,
        percent_removed: int = 10,
        k_sigma: int = 2,
        extended: int = 10,
        ghost2: bool = "HARPS03",
        borders_pxl: bool = False,
    ) -> None:
        return sts.outliers.yarara_correct_brute(
            self,
            sub_dico,
            continuum,
            reference,
            win_roll,
            min_length,
            percent_removed,
            k_sigma,
            extended,
            ghost2,
            borders_pxl,
        )

    def uncorrect_hole(
        self: spec_time_series,
        conti: ndarray,
        conti_ref: ndarray,
        values_forbidden: List[Union[int, float]] = [0, np.inf],
    ) -> ndarray:
        return sts.processing.uncorrect_hole(self, conti, conti_ref, values_forbidden)
