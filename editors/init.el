(require 'package)
(add-to-list 'package-archives '("marmalade" . "https://marmalade-repo.org/packages/") t)
(add-to-list 'package-archives '("tromey" . "http://tromey.com/elpa/") t)
(add-to-list 'package-archives '("melpam" . "http://melpa.milkbox.net/packages/") t)
(add-to-list 'package-archives '("melpa.org" . "http://melpa.org/packages/"))
;(add-to-list 'package-archives '("melpa" . "http://melpa-stable.milkbox.net/packages/"))
(add-to-list 'package-archives '("melpas" . "http://melpa-stable.milkbox.net/packages/") t)
;(add-to-list 'package-archives '("gnu" . "http://elpa.gnu.org/packages") t)
;
;; (setq package-archives '(("gnu" . "http://elpa.gnu.org/packages/")
;;                          ("marmalade" . "http://marmalade-repo.org/packages/")
;;                          ("melpa" . "http://melpa-stable.milkbox.net/packages/")))
(package-initialize)
(when (not package-archive-contents) (package-refresh-contents))

(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(ansi-color-faces-vector
   [default default default italic underline success warning error])
 '(display-battery-mode t)
 '(evil-visual-mark-mode t)
 '(inhibit-startup-screen t)
 '(package-selected-packages
   (quote
    (typescript-mode typescript pacmacs feature-mode gherkin-mode vimish-fold clj-refactor markdown-mode+ markdown-preview-mode kibit-helper buffer-move evil-numbers marmalade-client slime love-minor-mode lua-mode github-modern-theme github-theme darcula-theme imenu-list magit helm evil-leader which-key evil-paredit paredit rainbow-delimiters neotree evil-visual-mark-mode rainbow-mode evil)))
 '(size-indication-mode t)
 '(tool-bar-mode nil))
(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(default ((t (:family "Lucida Console" :foundry "outline" :slant normal :weight normal :height 98 :width normal))))
 '(rainbow-delimiters-depth-1-face ((t (:foreground "red" :height 1.1))))
 '(rainbow-delimiters-depth-2-face ((t (:foreground "orange" :height 1.09))))
 '(rainbow-delimiters-depth-3-face ((t (:foreground "yellow" :height 1.07))))
 '(rainbow-delimiters-depth-4-face ((t (:foreground "green" :height 1.05))))
 '(rainbow-delimiters-depth-5-face ((t (:foreground "blue" :height 1.0))))
 '(rainbow-delimiters-depth-6-face ((t (:foreground "violet" :height 1.0))))
 '(rainbow-delimiters-depth-7-face ((t (:foreground "purple" :height 1.0))))
 '(rainbow-delimiters-depth-8-face ((t (:foreground "black" :height 0.9))))
 '(rainbow-delimiters-unmatched-face ((t (:background "cyan" :height 1.0)))))


(add-to-list 'default-frame-alist '(fullscreen . maximized))

(setq lua-default-application "c:/moje/progs/lua/lua.exe") 
;;(setq diff-path "c:/moje/progs/diff3/bin") ;;no effect
(setq ediff-diff3-program "c:/moje/progs/diff3/bin/diff3.exe") ;;stejne jeste nekde nastavit vertikalni diff!!!
;                c:\moje\progs\Vim\vim74\diff.exe 

(setq slime-default-lisp 'ccl)
(setq inferior-lisp-program "c:/moje/progs/ccl/wx86cl64.exe")
;;(setq inferior-lisp-program "sbcl")
;;(setq inferior-lisp-program "clisp")

(require 'evil)
;(require 'evil-leader)
(evil-mode 1)
(evil-visual-mark-mode)
;(global-evil-leader-mode)
;(evil-)
(setq evil-flash-delay 7)
(evil-select-search-module 'evil-search-module 'evil-search)


(rainbow-mode)
(require 'rainbow-delimiters)
(add-hook 'prog-mode-hook 'rainbow-delimiters-mode)
;(add-hook 'prog-mode-hook 'rainbow-delimiters-mode)
;(require 'rainbow-delimiters)

;(set-face-attribute 'rainbow-delimiters-unmatched-face nil
;                    :foreground 'unspecified
;                    :inherit 'error)
;; rainbow-delimiters-mode setup, with decreasing bracket size
;(custom-set-faces
  ;;; custom-set-faces was added by Custom.
  ;;; If you edit it by hand, you could mess it up, so be careful.
  ;;; Your init file should contain only one such instance.
  ;;; If there is more than one, they won't work right.

  ;'(rainbow-delimiters-depth-1-face ((t (:foreground "red" :height 2.0))))
  ;'(rainbow-delimiters-depth-2-face ((t (:foreground "orange" :height 1.8))))
  ;'(rainbow-delimiters-depth-3-face ((t (:foreground "yellow" :height 1.6))))
  ;'(rainbow-delimiters-depth-4-face ((t (:foreground "green" :height 1.4))))
  ;'(rainbow-delimiters-depth-5-face ((t (:foreground "blue" :height 1.2))))
  ;'(rainbow-delimiters-depth-6-face ((t (:foreground "violet" :height 1.1))))
  ;'(rainbow-delimiters-depth-7-face ((t (:foreground "purple" :height 1.0))))
  ;'(rainbow-delimiters-depth-8-face ((t (:foreground "black" :height 0.9))))
  ;'(rainbow-delimiters-unmatched-face ((t (:background "cyan" :height 0.8))))
  ;)


;;no question about activ proccesses
(require 'cl)
(defadvice save-buffers-kill-emacs (around no-query-kill-emacs activate)
           (cl-flet ((process-list ())) ad-do-it))

;;send to REPL C-;
(require 'which-key)
(which-key-mode)
(defun cider-eval-expression-at-point-in-repl ()
  (interactive)
  (let ((form (cider-defun-at-point)))
    ;; Strip excess whitespace
    (while (string-match "\\`\s+\\|\n+\\'" form)
           (setq form (replace-match "" t t form)))
    (set-buffer (cider-get-repl-buffer))
    (goto-char (point-max))
    (insert form)
    (cider-repl-return)))
(require 'cider-mode)
(define-key cider-mode-map
            (kbd "C-;") 'cider-eval-expression-at-point-in-repl)

;;(setq make-backup-files nil) ; stop creating backup~ files
;;(setq auto-save-default nil) ; stop creating #autosave# files
(setq backup-directory-alist '(("" . "~/.emacs.d/emacs-backup")))

;;verze nefunguji spolu uz zase 
;;DALSI CAS ztraceny tim abych to nejak rozchodil!!!!
;;a krome toho to pada i kdyz jsem to vyhodil: asi staci mit nainstalovane clj-refator a to to shodi .10.0-Spanshot jak to zmenit??? jinou veriz cideu???
;;
;;dalsi verze refactaoru mozna 2.4.1snapshotop
(require 'clj-refactor)
(defun my-clojure-mode-hook ()
    (clj-refactor-mode 1)
    (yas-minor-mode 1) ; for adding require/use/import statementsThis choice of keybinding leaves cider-macroexpand-1 unbound
    (cljr-add-keybindings-with-prefix "C-c C-m"))
(add-hook 'clojure-mode-hook #'my-clojure-mode-hook)


(defun save-all ()
  (interactive)
  (save-some-buffers t))


;;at to  aspon nepiska kdyz uz to neumi delat co ja chci a co ma!!!
(setq visible-bell 1)
(setq evil-move-beyond-eol t)

(global-set-key "\C-c*" 'text-scale-adjust)
(global-set-key "\C-ca" 'org-agenda)
(global-set-key "\C-cc" 'org-capture)
(global-set-key "\C-cL" 'org-store-link)  ;;snad se toto s nicim nepotluce

;;(setq org-todo-keywords
;;     '((sequence "TODO" "FEEDBACK" "VERIFY" "|" "DONE" "DELEGATED")))

;;org-log-into-drawer TODO state change
;;
(setq org-default-notes-file (concat org-directory "/notes.org"))
;;(define-key global-map "\C-cc" 'org-capture)
(setq org-capture-templates
      '(("t" "Todo" entry (file+headline "~/org/gtd.org" "Tasks")
         "* TODO %?\n %i\n %a")
        ("j" "Journal" entry (file+datetree "~/org/journal.org")
         "* %?\nEntered on %U\n %i\n %a")))

;;save when focus out
(add-hook 'focus-out-hook 'save-all)
(global-set-key (kbd "<f2>") 'save-buffer)
(global-set-key (kbd "C-c l") 'linum-mode)
(global-set-key (kbd "C-c m") 'evil-visual-mark-mode)
(global-set-key (kbd "C-c p") 'paredit-mode)
(global-set-key (kbd "C-c i") 'imenu-list-smart-toggle)
(global-set-key (kbd "C-c n") 'neotree-toggle)
(global-set-key (kbd "C-c w") 'superword-mode) ; doma mi to nefunguje!!!
(global-set-key (kbd "C-c M-g") 'magit-status)
;;(global-set-key (kbd "C-x M-g") 'magit-status+popus) ASI stare!!!
;;
;(global-set-key (kbd "C-c /") (evil-ex "s/\\\\///g<CR>"))
(global-set-key (kbd "C-c /") '(evil-ex "s/\\\\//g<CR>"))


